import os
import re
import csv
import json
import time
import shutil
import random
import hashlib
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
    # SciFact: GitHub raw might work; if not, clone repo.
    # File path you used is correct for that repo, but raw can still be blocked by ISP / 403 / TLS.
    try:
        p = strategy_github_raw(
            "scifact",
            "https://raw.githubusercontent.com/allenai/scifact/main/data/claims_train.jsonl",
            "claims_train.jsonl",
        )
    except Exception as e:
        log(f"   -> raw failed, fallback clone: {e}")
        p = strategy_git_clone(
            "scifact",
            "https://github.com/allenai/scifact.git",
            branch="main",
            wanted_relpath="data/claims_train.jsonl",
        )
    claims = parse_jsonl_extract(p, "claim")
    save_to_local("scifact", claims)

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
    FEVER dataset is often distributed via shared drives / not always in a clean GH raw.
    Your URL might break depending on repo state or LFS or permissions.
    We'll try:
      1) github raw mirror you used
      2) clone mirror repo
      3) (optional) HF snapshot if you have a repo id (fill in if you know)
    """
    # 1) raw
    try:
        p = strategy_github_raw(
            "fever",
            "https://raw.githubusercontent.com/eunsol/fever/master/data/fever/train.jsonl",
            "train.jsonl",
        )
        claims = parse_jsonl_extract(p, "claim")
        save_to_local("fever", claims)
        return
    except Exception as e:
        log(f"   -> raw failed: {e}")

    # 2) clone
    try:
        p = strategy_git_clone(
            "fever",
            "https://github.com/eunsol/fever.git",
            branch="master",
            wanted_relpath="data/fever/train.jsonl",
        )
        claims = parse_jsonl_extract(p, "claim")
        save_to_local("fever", claims)
        return
    except Exception as e:
        log(f"   -> clone failed: {e}")

    # 3) HF snapshot (you need to set the correct repo_id)
    # Example placeholder: replace with actual HF dataset repo if you use one
    # p = strategy_hf_snapshot("fever", "fever-dataset-repo-id", ["**/*train*.jsonl"])
    raise RuntimeError("FEVER download failed via raw+clone. Provide a HuggingFace repo_id you want to use as source.")

def download_hover():
    """
    HoVer often large and may be LFS; try raw, then clone, then HF snapshot (if repo_id known).
    """
    try:
        p = strategy_github_raw(
            "hover",
            "https://raw.githubusercontent.com/hummer-hover/hover/master/data/hover_train_release_v1.1.jsonl",
            "hover_train_release_v1.1.jsonl",
        )
    except Exception as e:
        log(f"   -> raw failed, fallback clone: {e}")
        p = strategy_git_clone(
            "hover",
            "https://github.com/hummer-hover/hover.git",
            branch="master",
            wanted_relpath="data/hover_train_release_v1.1.jsonl",
        )

    claims = parse_jsonl_extract(p, "claim")
    save_to_local("hover", claims)

# ---------- main ----------
def download_all():
    log("🚀 Starting Robust Benchmark Downloader ...")
    tasks: List[Callable[[], None]] = [
        download_truthfulqa,
        download_scifact,
        download_climate_fever,
        download_hover,
        download_fever,  # keep last because it may fail depending on source
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
