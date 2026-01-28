
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Define paths
RAW_DIR = Path("data/benchmarks_raw")
PROCESSED_DIR = Path("data/benchmarks_processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def process_fever():
    """
    Process FEVER raw data (train.jsonl) into a labeled format.
    Items with 'NOT ENOUGH INFO' are often excluded or kept depending on use case.
    We will keep all but normalize format.
    
    Target Format:
    [
      {
        "id": ...,
        "claim": "...",
        "label": "SUPPORTS" | "REFUTED" | "NOT ENOUGH INFO",
        "evidence": ...
      },
      ...
    ]
    """
    print("\nProcessing FEVER...")
    raw_path = RAW_DIR / "fever" / "train.jsonl"
    
    if not raw_path.exists():
        log(f"❌ Raw file not found: {raw_path}. Run download_data.py first.")
        return

    processed_data = []
    
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                # maintain essential fields
                item = {
                    "id": obj.get("id"),
                    "claim": obj.get("claim"),
                    "label": obj.get("label"),
                    # evidence can be complex in FEVER, keeping raw for now
                    "evidence": obj.get("evidence") 
                }
                processed_data.append(item)
            except json.JSONDecodeError:
                pass
                
    out_path = PROCESSED_DIR / "fever_labeled.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    log(f"✅ FEVER processed: {len(processed_data)} items -> {out_path}")
    
    # Also create a simple mapping for quick lookups if needed
    # (Optional)

def process_copheme():
    """
    Process CoPHEME data (Lying with Truths paper version).
    Structure:
      data/benchmarks_raw/copheme/[event]/[event]_hypotheses.json
    
    Each item has:
     - tweet_id
     - text (The claim/tweet content)
     - conclusion (The derived claim conclusion)
     - veracity (false, unverified, true?)
    """
    print("\nProcessing CoPHEME...")
    base_dir = RAW_DIR / "copheme"
    
    if not base_dir.exists():
        log(f"❌ CoPHEME data not found at {base_dir}. Run download_data.py first.")
        return

    processed_data = []
    
    # Iterate over event folders
    for event in os.listdir(base_dir):
        event_path = base_dir / event
        if not event_path.is_dir(): continue
        
        # Look for hypotheses file
        hypo_file = event_path / f"{event}_hypotheses.json"
        if not hypo_file.exists():
             # maybe named differently? listing files
             candidates = list(event_path.glob("*_hypotheses.json"))
             if candidates:
                 hypo_file = candidates[0]
             else:
                 continue

        try:
            with open(hypo_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    # Map to standard format
                    # Using 'conclusion' as the main claim often makes sense for fact-checking
                    # BUT 'text' is the raw tweet. User usually wants to check the claim.
                    # In CoPHEME, 'conclusion' is often the rumor being checked.
                    # Let's verify: 
                    # "text": "1st person killed ...", "conclusion": "Ahmed Merabet was the first victim..."
                    # We will store both if possible, or use 'conclusion' as primary claim text?
                    # Generally for fact checking benchmarks (like FEVER), 'claim' is a sentence.
                    # 'text' is the context/evidence.
                    # However, here 'text' is the source tweet containing the claim.
                    # I will use 'conclusion' as the 'claim' because it is a clean proposition.
                    
                    veracity = item.get("veracity", "unverified")
                    # normalize label
                    if veracity.lower() == "true": label = "SUPPORTS"
                    elif veracity.lower() == "false": label = "REFUTED"
                    elif veracity.lower() == "unverified": label = "NOT ENOUGH INFO"
                    else: label = veracity.upper()
                    
                    processed_data.append({
                        "id": item.get("tweet_id", str(item.get("timestamp"))),
                        "claim": item.get("conclusion"), 
                        "original_text": item.get("text"),
                        "label": label,
                        "topic": event,
                        "raw_veracity": veracity,
                        "evidence": [] # CoPHEME has separate evidence files, omitting for simple list
                    })
        except Exception as e:
            log(f"   ⚠️ Error processing {hypo_file}: {e}")

    if not processed_data:
        log("   ⚠️  No claims found. Check directory structure.")
    else:
        out_path = PROCESSED_DIR / "copheme_labeled.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        log(f"✅ CoPHEME processed: {len(processed_data)} items -> {out_path}")

def process_hover():
    """
    Process HoVer data.
    HoVer: Multi-hop reasoning.
    Raw file: data/benchmarks_raw/hover/hover_train_release_v1.1.json
    Labels: 'SUPPORTED', 'NOT_SUPPORTED' (equivalent to REFUTED?).
    We will map:
      'SUPPORTED' -> 'SUPPORTS'
      'NOT_SUPPORTED' -> 'REFUTED'
    """
    print("\nProcessing HoVer...")
    raw_path = RAW_DIR / "hover" / "hover_train_release_v1.1.json"
    
    if not raw_path.exists():
        log(f"❌ Raw file not found: {raw_path}. Run download_data.py first.")
        return

    processed_data = []
    
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            
        for obj in items:
            label_map = {
                "SUPPORTED": "SUPPORTS",
                "NOT_SUPPORTED": "REFUTED"
            }
            
            # HoVer items have: uid, claim, label, supporting_facts, ...
            raw_label = obj.get("label")
            label = label_map.get(raw_label, "NOT ENOUGH INFO") 
            
            processed_data.append({
                "id": str(obj.get("uid")),
                "claim": obj.get("claim"),
                "label": label,
                "evidence": obj.get("supporting_facts") 
            })
            
    except Exception as e:
        log(f"   ⚠️ Error processing HoVer: {e}")
        return
                
    out_path = PROCESSED_DIR / "hover_labeled.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    log(f"✅ HoVer processed: {len(processed_data)} items -> {out_path}")

def process_scifact():
    """
    Process SciFact data.
    Raw file: data/benchmarks_raw/scifact/claims_train.jsonl
    SciFact labels are often in 'evidence' object. 
    Format:
     {
       "id": 1,
       "claim": "...",
       "evidence": { "doc_id": [ {"label": "SUPPORT"|"CONTRADICT" ... } ] }
     }
    If no evidence supports/contradicts, it might be NEI?
    Actually SciFact train set usually has labels.
    Let's check structure. Typically it lists valid evidence.
    If 'evidence' is empty -> NOT ENOUGH INFO?
    However, often SciFact provides a consensus label or we derive it.
    Let's look at how typical SciFact loaders do it.
    It seems each evidence sentence has a label. We need to aggregate?
    """
    print("\nProcessing SciFact...")
    raw_path = RAW_DIR / "scifact" / "claims_train.jsonl"
    
    if not raw_path.exists():
        log(f"❌ Raw file not found: {raw_path}. Run download_data.py first.")
        return

    processed_data = []
    
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                # content: id, claim, evidence
                # Evidence is dict: "doc_id": [ { "sentences": [i], "label": "SUPPORT" } ]
                
                evidence_map = obj.get("evidence", {})
                
                final_label = "NOT ENOUGH INFO"
                
                # Simple aggregation: if any SUPPORT -> SUPPORTS. If any CONTRADICT -> REFUTED.
                # If both? Ambiguous. Usually datasets are clean.
                
                found_support = False
                found_contradict = False
                
                all_evidence = []
                
                for doc_id, proofs in evidence_map.items():
                    for proof in proofs:
                        lbl = proof.get("label")
                        if lbl == "SUPPORT": found_support = True
                        if lbl == "CONTRADICT": found_contradict = True
                        all_evidence.append(proof)
                        
                if found_support and not found_contradict:
                    final_label = "SUPPORTS"
                elif found_contradict and not found_support:
                    final_label = "REFUTED"
                elif found_support and found_contradict:
                    final_label = "NOT ENOUGH INFO" # Conflict?
                else: 
                     final_label = "NOT ENOUGH INFO"
                     
                processed_data.append({
                    "id": str(obj.get("id")),
                    "claim": obj.get("claim"),
                    "label": final_label,
                    "evidence": all_evidence
                })
            except json.JSONDecodeError:
                pass
                
    out_path = PROCESSED_DIR / "scifact_labeled.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    log(f"✅ SciFact processed: {len(processed_data)} items -> {out_path}")

def main():
    process_fever()
    process_copheme()
    process_hover()
    process_scifact()

if __name__ == "__main__":
    main()
