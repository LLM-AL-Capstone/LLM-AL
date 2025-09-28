from pathlib import Path
import yaml, json, pandas as pd

TASKS_DIR = Path("configs/tasks")

def load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).read_text())

def load_task_cfg(task_name: str):
    return yaml.safe_load((TASKS_DIR / f"{task_name}.yaml").read_text())

def load_splits(task_cfg):
    tr = pd.read_csv(task_cfg["split"]["train"])
    te = pd.read_csv(task_cfg["split"]["test"])
    return tr, te

def write_json(path: str | Path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return p
