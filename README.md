# Counterfactual-Driven Few-shot Labeling (LLM-only)

**Purpose**: Produce a **dummy baseline** using only open-source LLMs:
- Generate counterfactuals (CFs) from TRAIN
- Filter using **LLM-only** (annotator + filter)
- Pick top **k demos**
- Few-shot label **TEST** and report metrics (Acc / Macro-F1)

> It uses **Ollama** to run compact models locally (e.g., `qwen:7b-chat`).

---

## Quickstart

1) **Install Ollama** and pull a small instruct model:
   ```bash
   ollama pull qwen:7b-chat
   ```

2) **Create a Python env** and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3) **Place your data** under `data/splits/` (see samples).

4) **Generate counterfactual demos** from TRAIN (targets ~120 filtered sentences):
   ```bash
   make demos TASK=emotions SAMPLE=500
   # Note: Default task is 'yelp' but 'emotions' is recommended for testing
   ```

5) **Few-shot label TEST** and evaluate:
   ```bash
   make eval TASK=emotions
   ```

6) **Multi-shot evaluation** (test with 10, 15, 30, 50, 70, 90, 120 demos):
   ```bash
   make multi-eval TASK=emotions
   ```

7) Results in `reports/runs/` with detailed multi-shot metrics.

---

## Generated Files

The pipeline creates several output files:

**Demo Generation:**
- **`reports/demos/all_candidates_{task}_{model}_{params}_{timestamp}.json`** – All candidates from each run
- **`reports/demos/demos_{task}_{model}_{params}_{timestamp}.json`** – Top-K demos from each run
- **`reports/demos/*_latest.json`** – Symlinks to most recent run (for convenience)

**Evaluation:**
- **`reports/runs/eval_{task}_{model}_{demo_params}_{timestamp}.json`** – Evaluation results per demo file

**Experimentation:**
```bash
# Generate demos with different parameters (preserves all runs)
make demos TASK=emotions K=5 SAMPLE=100
make demos TASK=emotions K=10 SAMPLE=200

# Evaluate with latest demos
make eval TASK=emotions

# Evaluate with specific demo file  
make eval TASK=emotions DEMOS=demos_emotions_qwen2.5_1.5b-instruct_s100_k5_20250928_123456.json

# Select different K from saved candidates
python -m src.app.services.demos.select_top_k --task emotions --k 15

# View generated counterfactuals
python scripts/view_cf.py --file reports/demos/all_candidates_emotions_latest.json --limit 5
```

## New Features

### Multi-Shot Evaluation
Test performance with different few-shot counts (10, 15, 30, 50, 70, 90, 120):
```bash
make multi-eval TASK=emotions
```

This outputs a table similar to research papers showing:
- Macro F1-scores for each few-shot count
- Accuracy scores for each few-shot count
- Uses the top-K scored candidates from filtering

### Viewing Generated Counterfactuals
```bash
# View latest generated CFs
python scripts/view_cf.py

# View specific file with filtering
python scripts/view_cf.py --file reports/demos/all_candidates_emotions_qwen_7b-chat*.json --min-score 0.8 --limit 10

# List all available candidate files
python scripts/view_cf.py --list-files
```

---

## Project layout
```
configs/              # YAML configs (global + per-task)
prompts/              # LLM prompts (generator, annotator, filter, annotator_with_demos)
data/                 # your CSVs live here
reports/              # generated demos & evaluation metrics
  demos/              # counterfactual candidates & selected demos  
  runs/               # evaluation results & performance metrics
scripts/              # convenience scripts (optional)
src/app               # application code (modular services)
  llm/                # Ollama client + retries
  services/demos/     # make_demos (Plan A)
  services/eval/      # label_test (Plan A)
  filter/             # LLM filter
  generate/           # counterfactual generator
  annotate/           # annotator
  select/             # (reserved for AL variants)
  train/              # (reserved for training variants)
  utils/              # logging, IO, text helpers
```

---

## Commands

- `make install` – install dependencies
- `make demos TASK=yelp K=10 SAMPLE=100` – generate counterfactual candidates and select top-K demos
- `make eval TASK=yelp` – few-shot label TEST with latest demos and evaluate
- `make eval TASK=yelp DEMOS=specific_demo_file.json` – evaluate with specific demo file
- `make lint` / `make format` – code quality
- `make clean` – remove run artifacts

---

## Data schema

- **Train/Test CSV**: columns `text`, `label`

See `data/splits/*.csv` for examples.

---

## Configuration knobs (`configs/poc.yaml`)

```yaml
runner: ollama
model_gen: qwen:7b-chat
model_ann: qwen:7b-chat
temperature: 0.0
filter_threshold: 0.50
filter_max_new: 128
filter_target: 120

demo_generation:
  demo_count: 10
  demo_sample: 500
  diversity_cos_max: 0.90
  demos_path: reports/demos/demos_{task}.json
```

---

## Notes

- This is a **pure LLM-only** baseline (no hand-crafted Variation Theory rules, no classifier gating).
- Keep the same train/test split for all methods you compare.
- For stability, keep temperature at 0.0 for deterministic outputs and keep outputs short (8–64 tokens).

