import os
import re
import csv
import json
import time
import shutil
import random
import hashlib
import zipfile
import tarfile
import subprocess
from pathlib import Path
from typing import Optional, Callable, List, Any, Dict

import requests
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

SAVE_DIR = Path("data/benchmarks")
RAW_DIR = Path("data/benchmarks_raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------- utils ----------
def log(msg: str):
    print(msg, flush=True)

def clean_list(data_list: List[Any]) -> List[str]:
    clean_data = []
    for item in data_list:
        if item is None:
            continue
        s = str(item).strip()
        if len(s) > 10:
            clean_data.append(s)
    # de-dup while preserving order
    seen = set()
    out = []
    for s in clean_data:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def save_to_local(name: str, data_list: List[Any]):
    clean_data = clean_list(data_list)
    path = SAVE_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    log(f"✅ [SUCCESS] {name:<15}: {len(clean_data)} items saved -> {path}")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def requests_session() -> requests.Session:
    s = requests.Session()
    # Một số mạng/ISP chặn, UA giúp đỡ bị 403
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "*/*",
    })
    return s

def download_file_http(url: str, out_path: Path, timeout=30, max_retries=6) -> Path:
    """
    HTTP download with resume + retries + backoff.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sess = requests_session()

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"

            r = sess.get(url, stream=True, timeout=timeout, headers=headers)
            # GitHub sometimes returns HTML error page with status 200; detect it
            ct = (r.headers.get("Content-Type") or "").lower()

            if r.status_code in (403, 429):
                raise RuntimeError(f"HTTP {r.status_code} rate/forbidden")
            r.raise_for_status()

            total = r.headers.get("Content-Length")
            total = int(total) if total is not None else None

            mode = "ab" if resume_pos > 0 else "wb"
            with open(tmp_path, mode) as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}", initial=resume_pos
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Basic sanity: avoid saving an HTML error page
            if out_path.suffix.lower() in [".json", ".jsonl", ".csv", ".txt"]:
                with open(tmp_path, "rb") as f:
                    head = f.read(200).lstrip()
                if head.startswith(b"<!DOCTYPE html") or head.startswith(b"<html"):
                    raise RuntimeError("Downloaded HTML instead of data (likely blocked / wrong URL).")

            tmp_path.replace(out_path)
            return out_path

        except Exception as e:
            wait = min(60, (2 ** attempt) + random.random() * 2)
            log(f"⚠️  Download failed (attempt {attempt}/{max_retries}) for {url}: {e}")
            log(f"    -> backoff {wait:.1f}s")
            time.sleep(wait)

            # refresh resume_pos
            resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0

    raise RuntimeError(f"Failed to download after {max_retries} retries: {url}")

def run(cmd: List[str], cwd: Optional[Path] = None):
    log("   $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def git_clone_or_pull(repo_url: str, dst: Path, branch: Optional[str] = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (dst / ".git").exists():
        run(["git", "fetch", "--all", "--prune"], cwd=dst)
        if branch:
            run(["git", "checkout", branch], cwd=dst)
        run(["git", "pull"], cwd=dst)
    else:
        cmd = ["git", "clone"]
        if branch:
            cmd += ["-b", branch]
        cmd += [repo_url, str(dst)]
        run(cmd)

def git_lfs_pull(dst: Path):
    # If repo uses LFS, this will fetch actual large files
    run(["git", "lfs", "pull"], cwd=dst)

# ---------- parsers ----------
def parse_jsonl_extract(path: Path, key: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if key in obj:
                out.append(obj[key])
    return out

def parse_csv_column(path: Path, column_name: Optional[str] = None, column_index: Optional[int] = None) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return out
    header = rows[0]
    if column_name is not None:
        try:
            idx = header.index(column_name)
        except ValueError:
            raise RuntimeError(f"Column '{column_name}' not found. Available: {header[:20]}")
    elif column_index is not None:
        idx = column_index
    else:
        idx = 0
    for row in rows[1:]:
        if len(row) > idx:
            out.append(row[idx])
    return out

# ---------- HF snapshot helper ----------
def hf_snapshot(repo_id: str, local_dir: Path, allow_patterns: Optional[List[str]] = None, revision: str = "main") -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not installed. pip install huggingface_hub")
    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"   ⏳ HF snapshot: {repo_id} (patterns={allow_patterns})")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        # If you are rate-limited, set your HF token in env: HUGGINGFACE_HUB_TOKEN
    )
    return local_dir

# ---------- dataset download strategies ----------
def strategy_github_raw(name: str, url: str, out_rel: str) -> Path:
    log(f"   ⏳ GitHub RAW: {name}")
    out_path = RAW_DIR / name / out_rel
    return download_file_http(url, out_path)

def strategy_git_clone(name: str, repo_url: str, branch: Optional[str], wanted_relpath: str) -> Path:
    log(f"   ⏳ Git clone: {name}")
    repo_dir = RAW_DIR / name / "repo"
    git_clone_or_pull(repo_url, repo_dir, branch=branch)
    # Try LFS pull (safe even if no LFS)
    try:
        git_lfs_pull(repo_dir)
    except Exception as e:
        log(f"   (note) git lfs pull failed/skip: {e}")

    target = repo_dir / wanted_relpath
    if not target.exists():
        # try search by basename
        base = Path(wanted_relpath).name
        matches = list(repo_dir.rglob(base))
        if matches:
            target = matches[0]
        else:
            raise RuntimeError(f"File not found in repo: {wanted_relpath}")
    out_path = RAW_DIR / name / target.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, out_path)
    return out_path

def strategy_hf_snapshot(name: str, repo_id: str, allow_patterns: List[str]) -> Path:
    log(f"   ⏳ HF snapshot: {name}")
    local_dir = RAW_DIR / name / "hf"
    hf_snapshot(repo_id, local_dir, allow_patterns=allow_patterns)
    # pick first matched file
    candidates = []
    for pat in allow_patterns:
        # simplistic glob
        candidates += list(local_dir.rglob(Path(pat).name))
    if not candidates:
        # fallback: any file with those extensions
        candidates = list(local_dir.rglob("*.jsonl")) + list(local_dir.rglob("*.json")) + list(local_dir.rglob("*.csv"))
    if not candidates:
        raise RuntimeError(f"No files found after HF snapshot for {repo_id}")
    out_path = RAW_DIR / name / candidates[0].name
    shutil.copy2(candidates[0], out_path)
    return out_path

def strategy_download_zip_and_extract(name: str, url: str) -> Path:
    """
    Download a zip/tar file, extract it, and return the directory.
    """
    log(f"   ⏳ Download & Extract: {name}")
    archive_path = RAW_DIR / name / "archive.file"
    download_file_http(url, archive_path)
    
    extract_dir = RAW_DIR / name / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Check file type
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=extract_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        # fallback
        pass

    log(f"   -> Extracted to {extract_dir}")
    return extract_dir

# ---------- dataset configs ----------
def download_truthfulqa():
    # TruthfulQA: CSV on GitHub is usually easy
    p = strategy_github_raw(
        "truthfulqa",
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        "TruthfulQA.csv",
    )
    # Column name in TruthfulQA csv: "Question" (commonly). If not found, fallback to column 0.
    try:
        questions = parse_csv_column(p, column_name="Question")
    except Exception:
        questions = parse_csv_column(p, column_index=0)
    save_to_local("truthfulqa", questions)

def download_scifact():
    """
    Download SciFact from official S3 bucket (as per repo script).
    """
    url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
    try:
        extract_dir = strategy_download_zip_and_extract("scifact", url)
        # The tar usually extracts to a 'data' folder.
        # Check structure
        # data/claims_train.jsonl
        
        target_file = extract_dir / "data" / "claims_train.jsonl"
        if not target_file.exists():
            # maybe it extracted directly?
            target_file = extract_dir / "claims_train.jsonl"
            
        if not target_file.exists():
             # Recursive search
             candidates = list(extract_dir.rglob("claims_train.jsonl"))
             if candidates:
                 target_file = candidates[0]
             else:
                 raise RuntimeError("claims_train.jsonl not found in SciFact archive")
        
        claims = parse_jsonl_extract(target_file, "claim")
        save_to_local("scifact", claims)
        
        # Also copy the raw file to a known location for processing
        shutil.copy2(target_file, RAW_DIR / "scifact" / "claims_train.jsonl")
        
    except Exception as e:
        log(f"   ❌ SciFact download failed: {e}")

def download_climate_fever():
    # Many mirrors; if raw fail, clone.
    try:
        p = strategy_github_raw(
            "climate_fever",
            "https://raw.githubusercontent.com/asahi417/climate-fever/main/data/climate_fever.jsonl",
            "climate_fever.jsonl",
        )
    except Exception as e:
        log(f"   -> raw failed, fallback clone: {e}")
        p = strategy_git_clone(
            "climate_fever",
            "https://github.com/asahi417/climate-fever.git",
            branch="main",
            wanted_relpath="data/climate_fever.jsonl",
        )
    claims = parse_jsonl_extract(p, "claim")
    save_to_local("climate_fever", claims)

def download_fever():
    """
    Download FEVER dataset (approx 185k claims).
    We try official fever.ai links first.
    """
    # 1) Official
    try:
        url = "https://fever.ai/download/fever/train.jsonl"
        p = strategy_github_raw("fever", url, "train.jsonl")
        claims = parse_jsonl_extract(p, "claim")
        save_to_local("fever", claims)
        return
    except Exception as e:
        log(f"   -> official failed: {e}")

    # 2) Fallback to HF if official fails
    try:
        # This requires huggingface_hub
        # We need a specific good repo. 'fever/fever' might not be a valid repo id for direct snapshot.
        # But let's try a known mirror if exists.
        # For now, we will rely on official link mostly.
        pass
    except Exception as e:
        log(f"   -> HF failed: {e}")

    raise RuntimeError("FEVER download failed. Please check internet connection or manually download from https://fever.ai/download/fever/train.jsonl")

def download_copheme():
    """
    Download CoPHEME dataset from 'Lying with Truths' paper repo.
    Repo: https://github.com/CharlesJW222/Lying_with_Truth
    """
    try:
        repo_url = "https://github.com/CharlesJW222/Lying_with_Truth.git"
        log("   ⏳ Git clone: CoPHEME (Lying_with_Truth)")
        repo_dir = RAW_DIR / "copheme_repo"
        git_clone_or_pull(repo_url, repo_dir, branch="main")
        
        # The data is in 'CoPHEME' folder inside repo
        src_data_dir = repo_dir / "CoPHEME"
        dst_data_dir = RAW_DIR / "copheme"
        
        if not src_data_dir.exists():
            raise RuntimeError(f"CoPHEME folder not found in {repo_dir}")
            
        if dst_data_dir.exists():
             shutil.rmtree(dst_data_dir)
             
        shutil.copytree(src_data_dir, dst_data_dir)
        log(f"   ✅ CoPHEME data ready at {dst_data_dir}")
        
    except Exception as e:
        log(f"   ❌ CoPHEME download failed: {e}")

def download_hover():
    """
    HoVer: download validation/train set from official site.
    URL: https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json
    Note: The file extension is .json, it is likely a JSON array, not JSONL.
    """
    try:
        url = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
        target_filename = "hover_train_release_v1.1.json"
        
        # Download
        out_path = RAW_DIR / "hover" / target_filename
        download_file_http(url, out_path)
        
        # Parse: HoVer JSON is a list of objects
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # extract claims
        claims = []
        for item in data:
            if "claim" in item:
                claims.append(item["claim"])
                
        save_to_local("hover", claims)
        
    except Exception as e:
        log(f"   -> official download failed: {e}")
        raise e

# ---------- main ----------
def download_all():
    log("🚀 Starting Robust Benchmark Downloader ...")
    tasks: List[Callable[[], None]] = [
        download_truthfulqa,
        download_scifact,
        download_climate_fever,
        download_hover,
        download_fever,
        download_copheme,
    ]

    failed = []
    for fn in tasks:
        name = fn.__name__.replace("download_", "")
        log(f"\n=== {name.upper()} ===")
        try:
            fn()
        except Exception as e:
            log(f"❌ {name} FAILED: {e}")
            failed.append((name, str(e)))

    log("\n========================")
    if failed:
        log("Some datasets failed:")
        for n, err in failed:
            log(f" - {n}: {err}")
        log("Tip: If FEVER fails, point to a specific HuggingFace repo_id or an accessible mirror.")
    else:
        log("✅ All datasets done!")

if __name__ == "__main__":
    download_all()
