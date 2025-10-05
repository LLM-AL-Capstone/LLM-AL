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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--demos", default=None, help="Specific demo file to use (default: latest)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    task_cfg = load_task_cfg(args.task)
    labels = task_cfg["labels"]
    test_df = pd.read_csv(task_cfg["split"]["test"])

    # Get field mappings
    text_field = task_cfg["fields"]["text"]
    label_field = task_cfg["fields"]["label"]

    # Use new config section name with backward compatibility
    demo_config = cfg.get("demo_generation", cfg.get("planA", {}))
    
    # Determine which demo file to use
    if args.demos:
        # Use specified demo file
        if args.demos.startswith("reports/demos/"):
            demos_path = args.demos
        else:
            demos_path = f"reports/demos/{args.demos}"
        demo_file_id = Path(args.demos).stem  # Extract filename without extension
    else:
        # Use latest demos file (symlink to most recent run) 
        demos_path = f"reports/demos/demos_{args.task}_latest.json"
        # Resolve symlink to get actual filename for report naming
        resolved_path = Path(demos_path).resolve()
        demo_file_id = resolved_path.stem
    
    demos = json.loads(Path(demos_path).read_text())

    tpl = "prompts/annotation/annotator_with_demos.txt"
    client = OllamaClient(cfg["model_ann"], temperature=0.2)

    preds = []
    for text in test_df[text_field].tolist():
        prompt = make_prompt(tpl, labels, demos, text)
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

    m = compute_metrics(test_df[label_field].tolist(), preds, labels)
    
    # Create descriptive filename with task, model, and demo info
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg["model_ann"].replace(":", "_").replace("/", "_")
    
    # Extract demo parameters from demo_file_id for cleaner naming
    if "demos_" in demo_file_id:
        demo_params = demo_file_id.replace("demos_", "").replace(f"{args.task}_", "")
    else:
        demo_params = "latest"
    
    filename = f"eval_{args.task}_{model_name}_{demo_params}_{timestamp}.json"
    
    # Add demo file info to metrics
    m["demo_file_used"] = demos_path
    m["num_demos"] = len(demos)
    m["eval_timestamp"] = timestamp
    
    write_json(f"reports/runs/{filename}", m)
    print(f"Test results saved to reports/runs/{filename}")
    print(f"Used {len(demos)} demos from: {demos_path}")
    print(f"{args.task.title()} emotion classification metrics:", m)

if __name__ == "__main__":
    main()
