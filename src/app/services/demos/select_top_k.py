#!/usr/bin/env python3
"""
Utility script to select top-K demos from saved candidates.
This allows experimenting with different K values without regenerating all candidates.
"""

import argparse
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.io import write_json

def cosine(u, v):
    import math
    num = sum(x*y for x,y in zip(u,v))
    du = math.sqrt(sum(x*x for x in u))
    dv = math.sqrt(sum(x*x for x in v))
    return num / max(1e-9, du*dv)

def main():
    ap = argparse.ArgumentParser(description="Select top-K demos from saved candidates")
    ap.add_argument("--task", required=True, help="Task name")
    ap.add_argument("--k", type=int, required=True, help="Number of demos to select")
    ap.add_argument("--diversity-threshold", type=float, default=0.9, 
                    help="Cosine similarity threshold for diversity filtering")
    args = ap.parse_args()

    # Load all candidates (use latest by default)
    candidates_path = f"reports/demos/all_candidates_{args.task}_latest.json"
    if not Path(candidates_path).exists():
        # Fallback to old naming if latest doesn't exist
        candidates_path = f"reports/demos/all_candidates_{args.task}.json"
        if not Path(candidates_path).exists():
            print(f"Error: No candidate files found. Run make_demos first.")
            return

    candidates = json.loads(Path(candidates_path).read_text())
    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    # Apply diversity filter and select top-K
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = embedder.encode([c["counterfactual"] for c in candidates], normalize_embeddings=True)
        print("Using sentence transformer for diversity filtering")
    except Exception:
        vecs = None
        print("Sentence transformer not available, skipping diversity filtering")

    selected, selected_idx = [], []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["score"], reverse=True)
    
    for i in order:
        if len(selected) >= args.k:
            break
        
        if vecs is not None:
            keep = True
            for jdx in selected_idx:
                if cosine(vecs[i], vecs[jdx]) > args.diversity_threshold:
                    keep = False
                    break
            if not keep: 
                continue
        
        selected.append(candidates[i])
        selected_idx.append(i)

    # Save selected demos with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"reports/demos/demos_{args.task}_k{args.k}_{timestamp}.json"
    write_json(output_path, selected)
    
    print(f"Selected {len(selected)} diverse demos (K={args.k})")
    print(f"Saved to {output_path}")
    
    if len(selected) < args.k:
        print(f"Warning: Only {len(selected)} demos available (requested {args.k})")

if __name__ == "__main__":
    main()