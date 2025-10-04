import argparse, json
from pathlib import Path
import yaml, pandas as pd
from jinja2 import Template
from ...utils.io import load_task_cfg, load_yaml, write_json
from ...llm.ollama import OllamaClient
from ...utils.io import load_yaml
from ...utils.io import load_task_cfg
from ...utils.io import write_json

def make_prompt(tpl_path: str, labels, demos, text):
    demo_lines = []
    for d in demos:
        demo_lines.append(f'Text: {d["counterfactual"]}\nLabel: {d["counterfactual_label"]}')
    demo_block = "\n\n".join(demo_lines)
    tpl = Template(Path(tpl_path).read_text())
    return tpl.render(labels=labels, demo_block=demo_block, text=text)

def compute_metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "n": len(y_true),
    }

def evaluate_with_k_demos(cfg, task_cfg, test_df, all_candidates, k, text_field, label_field):
    """Evaluate with top-k demos"""
    labels = task_cfg["labels"]
    
    # Select top-k demos based on score
    sorted_candidates = sorted(all_candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    top_k_demos = sorted_candidates[:k]
    
    if len(top_k_demos) < k:
        print(f"Warning: Only {len(top_k_demos)} demos available, requested {k}")
    
    tpl = "prompts/annotator_with_demos.txt"
    client = OllamaClient(cfg["model_ann"], temperature=0.2)

    preds = []
    test_texts = test_df[text_field].tolist()
    print(f"    Evaluating {len(test_texts)} test examples...")
    
    for i, text in enumerate(test_texts):
        if (i + 1) % 10 == 0 or i < 5:  # Show progress every 10 examples
            print(f"      Progress: {i+1}/{len(test_texts)} examples")
        
        prompt = make_prompt(tpl, labels, top_k_demos, text)
        out = client.run(prompt, system=None, max_tokens=cfg["ann_max_new"], retries=1)
        i, j = out.find("{"), out.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                obj = json.loads(out[i:j+1])
                preds.append(obj.get("label", labels[0]))
            except Exception:
                preds.append(labels[0])
        else:
            preds.append(labels[0])

    return compute_metrics(test_df[label_field].tolist(), preds, labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--candidates", default=None, help="Specific candidate file to use (default: latest)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    task_cfg = load_task_cfg(args.task)
    labels = task_cfg["labels"]
    test_df = pd.read_csv(task_cfg["split"]["test"])

    # Get field mappings
    text_field = task_cfg["fields"]["text"]
    label_field = task_cfg["fields"]["label"]

    # Determine which candidate file to use
    if args.candidates:
        # Use specified candidate file
        if args.candidates.startswith("reports/demos/"):
            candidates_path = args.candidates
        else:
            candidates_path = f"reports/demos/{args.candidates}"
    else:
        # Use latest candidates file (symlink to most recent run) 
        candidates_path = f"reports/demos/all_candidates_{args.task}_latest.json"
    
    all_candidates = json.loads(Path(candidates_path).read_text())
    
    # Test different few-shot counts
    few_shot_counts = [10, 15, 30, 50, 70, 90, 120]
    
    print(f"Loaded {len(all_candidates)} candidates from: {candidates_path}")
    print(f"Testing with few-shot counts: {few_shot_counts}")
    print(f"Using model: {cfg['model_ann']}")
    print(f"Test dataset size: {len(test_df)} examples")
    results = {}
    
    print(f"\nEvaluating {args.task.upper()} with different few-shot counts...")
    print("=" * 60)
    
    for k in few_shot_counts:
        if k <= len(all_candidates):
            print(f"\nEvaluating with {k} few-shot examples...")
            metrics = evaluate_with_k_demos(cfg, task_cfg, test_df, all_candidates, k, text_field, label_field)
            results[k] = metrics
            print(f"  Accuracy: {metrics['accuracy']:.3f}, Macro-F1: {metrics['macro_f1']:.3f}")
        else:
            print(f"\nSkipping {k} (not enough candidates: {len(all_candidates)})")
            results[k] = None

    # Print results table
    print("\n" + "="*80)
    print(f"MACRO F1-SCORES - {args.task.upper()}")
    print("="*80)
    print("Method        ", end="")
    for k in few_shot_counts:
        print(f"{k:>8}", end="")
    print()
    print("-" * 80)
    
    print("Counterfactuals ", end="")
    for k in few_shot_counts:
        if results[k] is not None:
            print(f"{results[k]['macro_f1']:.2f}    ", end="")
        else:
            print("  --    ", end="")
    print()
    
    print("\nACCURACY SCORES")
    print("-" * 80)
    print("Counterfactuals ", end="")
    for k in few_shot_counts:
        if results[k] is not None:
            print(f"{results[k]['accuracy']:.2f}    ", end="")
        else:
            print("  --    ", end="")
    print()

    # Save detailed results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg["model_ann"].replace(":", "_").replace("/", "_")
    
    filename = f"multi_shot_eval_{args.task}_{model_name}_{timestamp}.json"
    
    detailed_results = {
        "task": args.task,
        "model": cfg["model_ann"],
        "candidates_file": candidates_path,
        "total_candidates": len(all_candidates),
        "few_shot_results": results,
        "eval_timestamp": timestamp
    }
    
    write_json(f"reports/runs/{filename}", detailed_results)
    print(f"\nDetailed results saved to reports/runs/{filename}")

if __name__ == "__main__":
    main()