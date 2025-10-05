import argparse, json, random
from pathlib import Path
import yaml, pandas as pd
from ...utils.io import load_task_cfg, load_yaml, write_json
from ...generate.minimal_edit import generate_cf_with_patterns
from ...annotate.llm_annotator import annotate_label
from ...filter.variation_theory_filter import VariationTheoryFilter
from ...patterns.neurosymbolic_patterns import NeuroSymbolicPatternLearner

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
    ap.add_argument("--force-relearn", action="store_true", 
                   help="Force relearning patterns even if cache exists")
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
    
    # Also create symlink to "latest" for backward compatibility
    latest_candidates_path = f"reports/demos/all_candidates_{args.task}_latest.json"

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
    
    # STEP 3: Learn or Load Neuro-Symbolic Patterns
    pattern_learner = NeuroSymbolicPatternLearner(cfg)
    all_patterns = pattern_learner.learn_all_patterns(
        train_df, text_field, label_field, labels, 
        args.task, force_relearn=args.force_relearn
    )

    # Initialize Variation Theory Filter
    vt_filter = VariationTheoryFilter(cfg)

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
            # Randomized target label selection instead of round-robin
            available_labels = [label for label in labels if label != orig_label]
            to_label = random.choice(available_labels)

        generated_count += 1
        
        # Progress update every 10 attempts (since we're processing more)
        if generated_count % 10 == 0 or generated_count <= 5:
            print(f"Progress: {generated_count:3d}/{len(shuffled_train_df)} attempts | {len(candidates):3d} filtered")
        
        print(f"  Generating CF for: '{orig[:50]}{'...' if len(orig) > 50 else ''}' [{orig_label} -> {to_label}]")
        
        # STEP 4.1: Identify candidate phrases using learned patterns
        candidate_info = pattern_learner.identify_candidate_phrases(orig, all_patterns, orig_label)
        
        print(f"    Strategy: {candidate_info['strategy']}")
        if candidate_info['strategy'] == 'pattern_guided':
            print(f"    Pattern: {candidate_info['pattern_rule']}")
            print(f"    Modifiable: {candidate_info['phrases']}")
        
        # STEP 4.2 & 4.3: Generate multiple counterfactual candidates (OPTIMIZED: 2-3 instead of 3-5)
        try:
            cf_obj = generate_cf_with_patterns(cfg, task_cfg, orig, orig_label, to_label, candidate_info)
            cf_candidates = cf_obj.get("counterfactuals", [])
        except Exception as e:
            print(f"    Failed CF generation: {str(e)[:50]}...")
            empty_cf_count += 1
            continue

        if not cf_candidates:
            print(f"    No counterfactual candidates generated")
            empty_cf_count += 1
            continue

        print(f"    Generated {len(cf_candidates)} CF candidates")

        # STEP 5: Filter each candidate and select the best one (OPTIMIZED: Combined C2+C3 filter)
        best_candidate = None
        best_score = 0.0

        for idx, cf_candidate in enumerate(cf_candidates):
            cf_text = cf_candidate.get("text", "").strip()
            modification_focus = cf_candidate.get("modification_focus", "unknown")
            
            if not cf_text:
                continue
            
            print(f"      Candidate {idx+1}: '{cf_text[:40]}{'...' if len(cf_text) > 40 else ''}'")
            print(f"      Focus: {modification_focus}")
            
            # Verify label annotation
            ann = annotate_label(cfg, task_cfg, cf_text, labels)
            cf_label = ann.get("label", None)
            if cf_label != to_label:
                print(f"        Label mismatch: got '{cf_label}', expected '{to_label}'")
                continue

            print(f"        Label matches: '{cf_label}'")
            
            # Apply optimized Variation Theory filter (C1 + Combined C2+C3)
            vt_filter_result = vt_filter.apply_three_stage_filter(
                orig, cf_text, candidate_info, to_label
            )
            
            if vt_filter_result["pass_all"]:
                score = vt_filter_result["score"]
                pkr = vt_filter_result["pkr"]
                lfr = vt_filter_result["lfr"] 
                slfr = vt_filter_result["slfr"]
                print(f"        Passed VT filter! (score: {score:.3f}, PKR: {pkr:.3f}, LFR: {lfr:.3f}, SLFR: {slfr:.3f})")
                
                # Keep the best scoring candidate
                if score > best_score:
                    best_score = score
                    best_candidate = {
                        "text": cf_text,
                        "label": cf_label,
                        "score": score,
                        "modification_focus": modification_focus,
                        "filter": vt_filter_result["details"],
                        "pkr": pkr,
                        "lfr": lfr,
                        "slfr": slfr
                    }
            else:
                stage_failed = vt_filter_result["stage_failed"]
                reason = vt_filter_result["reason"]
                print(f"        Failed VT filter at stage {stage_failed}: {reason}")

        # Use the best candidate if found
        if best_candidate:
            print(f"    Best candidate (score: {best_candidate['score']:.3f}): '{best_candidate['text'][:50]}{'...' if len(best_candidate['text']) > 50 else ''}'")
            
            candidates.append({
                "original": orig,
                "original_label": orig_label,
                "counterfactual": best_candidate["text"],
                "counterfactual_label": best_candidate["label"],
                "score": best_candidate["score"],
                "filter": best_candidate["filter"],
                # NEW: Add pattern information
                "pattern_strategy": candidate_info['strategy'],
                "pattern_rule": candidate_info.get('pattern_rule', 'N/A'),
                "modifiable_parts": candidate_info.get('phrases', []),
                "modification_focus": best_candidate["modification_focus"]
            })
        else:
            print(f"    No candidates passed all filters")
            failed_filter_count += 1

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

    # Create symlink to "latest" file for easy access
    import os
    try:
        if os.path.exists(latest_candidates_path):
            os.remove(latest_candidates_path)
        os.symlink(os.path.basename(all_candidates_path), latest_candidates_path)
    except Exception as e:
        print(f"Note: Could not create symlink: {e}")
    
    print("\n" + "=" * 60)
    print(f"DEMO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"All {len(candidates)} candidates saved to: {all_candidates_path}")
    print(f"Latest file: {latest_candidates_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
