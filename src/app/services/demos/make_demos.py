import argparse, json, random
from pathlib import Path
import yaml, pandas as pd
from ...utils.io import load_task_cfg, load_yaml, write_json
from ...generate.minimal_edit import generate_cf
from ...annotate.llm_annotator import annotate_label
from ...filter.filter_llm import filter_llm

def cosine(u, v):
    import math
    num = sum(x*y for x,y in zip(u,v))
    du = math.sqrt(sum(x*x for x in u))
    dv = math.sqrt(sum(x*x for x in v))
    return num / max(1e-9, du*dv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, default=None)        # override demo_count
    ap.add_argument("--sample", type=int, default=None)   # override demo_sample
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    task_cfg = load_task_cfg(args.task)
    labels = task_cfg["labels"]

    # Use new config section name with backward compatibility
    demo_config = cfg.get("demo_generation", cfg.get("planA", {}))
    demo_k = args.k or demo_config.get("demo_count", 10)
    sample_n = args.sample or demo_config.get("demo_sample", 100)
    demos_path = demo_config.get("demos_path", f"prompts/demos_{args.task}.json").replace("{task}", args.task)
    diversity_cos_max = demo_config.get("diversity_cos_max", 0.9)
    
    # Create unique paths for this run including parameters
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg["model_gen"].replace(":", "_").replace("/", "_")
    
    # Generate unique filenames with run parameters
    run_id = f"{args.task}_{model_name}_s{sample_n}_k{demo_k}_{timestamp}"
    all_candidates_path = f"reports/demos/all_candidates_{run_id}.json"
    top_k_demos_path = f"reports/demos/demos_{run_id}.json"
    
    # Also create symlinks to "latest" for backward compatibility
    latest_candidates_path = f"reports/demos/all_candidates_{args.task}_latest.json"
    latest_demos_path = f"reports/demos/demos_{args.task}_latest.json"

    train_df, _ = (pd.read_csv(task_cfg["split"]["train"]), pd.read_csv(task_cfg["split"]["test"]))

    # Get field mappings
    text_field = task_cfg["fields"]["text"]
    label_field = task_cfg["fields"]["label"]

    # Target number of filtered sentences (configurable)
    filter_target = cfg.get("filter_target", 120)
    max_attempts = max(sample_n * 3, 1000)  # Prevent infinite loops
    
    # Sample and shuffle train data - LIMIT to sample_n
    shuffled_train_df = train_df.sample(n=min(sample_n, len(train_df)), random_state=42).reset_index(drop=True)
    
    print(f"Target: {filter_target} filtered candidates (will process all {len(shuffled_train_df)} examples)")
    print(f"Processing {len(shuffled_train_df)} training examples (no early stopping)")
    print(f"Using model: {cfg['model_gen']}")
    print("-" * 60)

    candidates = []
    generated_count = 0
    empty_cf_count = 0
    wrong_label_count = 0
    failed_filter_count = 0
    
    # Process all available training examples to get maximum candidates
    for i, row in shuffled_train_df.iterrows():
            
        orig = row[text_field]
        orig_label = row[label_field]
        
        # Skip rows with invalid labels
        if orig_label not in labels:
            print(f"  Skipping row with invalid label: '{orig_label}'")
            continue
            
        # choose target label
        if len(labels) == 2:
            to_label = labels[1] if orig_label == labels[0] else labels[0]
        else:
            idx = labels.index(orig_label)
            to_label = labels[(idx + 1) % len(labels)]

        generated_count += 1
        
        # Progress update every 10 attempts (since we're processing more)
        if generated_count % 10 == 0 or generated_count <= 5:
            print(f"Progress: {generated_count:3d}/{len(shuffled_train_df)} attempts | {len(candidates):3d} filtered")
        
        print(f"  Generating CF for: '{orig[:50]}{'...' if len(orig) > 50 else ''}' [{orig_label} -> {to_label}]")
        try:
            cf_obj = generate_cf(cfg, task_cfg, orig, orig_label, to_label)
            cf_text = cf_obj.get("counterfactual", "").strip()
        except Exception as e:
            print(f"    Failed CF generation: {str(e)[:50]}...")
            empty_cf_count += 1
            continue
            
        if not cf_text:
            print(f"    Empty counterfactual generated")
            empty_cf_count += 1
            continue

        print(f"    CF: '{cf_text[:50]}{'...' if len(cf_text) > 50 else ''}'")
        print(f"    Annotating label...")
        ann = annotate_label(cfg, task_cfg, cf_text, labels)
        cf_label = ann.get("label", None)
        if cf_label != to_label:
            print(f"    Label mismatch: got '{cf_label}', expected '{to_label}'")
            wrong_label_count += 1
            continue

        print(f"    Label matches: '{cf_label}'")
        print(f"    Applying filter...")
        j = filter_llm(cfg, orig, cf_text, to_label)
        if not j.get("pass_all", False):
            score = j.get("score", 0.0)
            print(f"    Failed filter (score: {score:.3f})")
            failed_filter_count += 1
            continue
        
        score = j.get("score", 0.0)
        print(f"    Passed filter! (score: {score:.3f})")

        candidates.append({
            "original": orig,
            "original_label": orig_label,
            "counterfactual": cf_text,
            "counterfactual_label": cf_label,
            "score": float(j.get("score", 0.0)),
            "filter": j.get("reasons", {})
        })

    print("\n" + "=" * 60)
    print(f"FILTER SUMMARY")
    print("=" * 60)
    print(f"Target candidates: {filter_target}")
    print(f"Successful candidates: {len(candidates)}")
    print(f"Total attempts: {generated_count}")
    print(f"Empty CFs: {empty_cf_count}")
    print(f"Wrong labels: {wrong_label_count}")
    print(f"Failed filter: {failed_filter_count}")
    print(f"Success rate: {len(candidates)/max(1, generated_count)*100:.1f}%")
    print("=" * 60)

    # Save all candidates for future K selection and multi-shot evaluation
    write_json(all_candidates_path, candidates)
    print(f"\nSaved {len(candidates)} raw candidates to: {all_candidates_path}")

    # Create a diverse subset for traditional demo selection (optional)
    print(f"\nCreating diverse demo subset (top {demo_k} diverse demos)...")
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Computing embeddings...")
        vecs = embedder.encode([c["counterfactual"] for c in candidates], normalize_embeddings=True)
    except Exception:
        print("Warning: Could not load sentence transformer, using score-only selection")
        vecs = None

    selected, selected_idx = [], []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["score"], reverse=True)
    print(f"Selecting top {demo_k} diverse demos from {len(candidates)} candidates...")
    
    for i in order:
        if len(selected) >= demo_k:
            break
        if vecs is not None:
            keep = True
            for jdx in selected_idx:
                if cosine(vecs[i], vecs[jdx]) > diversity_cos_max:
                    keep = False; break
            if not keep: 
                continue
        selected.append(candidates[i]); selected_idx.append(i)
        print(f"  Selected demo {len(selected)}: score {candidates[i]['score']:.3f}")

    print(f"\nSaving diverse demo subset...")
    write_json(top_k_demos_path, selected)
    
    # Create symlinks to "latest" files for easy access
    import os
    try:
        if os.path.exists(latest_candidates_path):
            os.remove(latest_candidates_path)
        if os.path.exists(latest_demos_path):
            os.remove(latest_demos_path)
        os.symlink(os.path.basename(all_candidates_path), latest_candidates_path)
        os.symlink(os.path.basename(top_k_demos_path), latest_demos_path)
    except Exception as e:
        print(f"Note: Could not create symlinks: {e}")
    
    print("\n" + "=" * 60)
    print(f"DEMO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Top {len(selected)} demos saved to: {top_k_demos_path}")
    print(f"All {len(candidates)} candidates: {all_candidates_path}")
    print(f"Latest files: {latest_demos_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
