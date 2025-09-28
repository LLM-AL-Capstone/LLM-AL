# Counterfactual-Driven Few-shot Labeling (LLM-only Plan A) — Industry-Grade POC

**Purpose**: Produce a **dummy baseline** using only open-source LLMs:
- Generate counterfactuals (CFs) from TRAIN
- Filter using **LLM-only** (annotator + judge)
- Pick top **5–10 demos**
- Few-shot label **TEST** and report metrics (Acc / Macro-F1)

> This repo is designed for a MacBook Air M3 (16 GB). It uses **Ollama** to run compact models locally (e.g., `qwen2.5:1.5b-instruct`).

---

## Quickstart

1) **Install Ollama** and pull a small instruct model:
   ```bash
   ollama pull qwen2.5:1.5b-instruct
   ```

2) **Create a Python env** and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3) **Place your data** under `data/splits/` (see samples).

4) **Generate counterfactual demos** from TRAIN:
   ```bash
   make demos TASK=yelp K=10 SAMPLE=100
   ```

5) **Few-shot label TEST** and evaluate:
   ```bash
   make eval TASK=yelp
   ```

6) Results in `reports/runs/test_run_{task}_{model}_metrics.json`.

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
```

---

## Project layout
```
configs/              # YAML configs (global + per-task)
prompts/              # LLM prompts (generator, annotator, judge, annotator_with_demos)
data/                 # your CSVs live here
reports/              # generated demos & evaluation metrics
  demos/              # counterfactual candidates & selected demos  
  runs/               # evaluation results & performance metrics
scripts/              # convenience scripts (optional)
src/app               # application code (modular services)
  llm/                # Ollama client + retries
  services/demos/     # make_demos (Plan A)
  services/eval/      # label_test (Plan A)
  judge/              # LLM judge
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
- **Unlabeled CSV (not used in Plan A)**: column `text`

See `data/splits/*.csv` for examples.

---

## Configuration knobs (`configs/poc.yaml`)

```yaml
runner: ollama
model_gen: qwen2.5:1.5b-instruct
model_ann: qwen2.5:1.5b-instruct
judge_threshold: 0.70
judge_max_new: 64

demo_generation:
  demo_count: 10
  demo_sample: 100
  diversity_cos_max: 0.90
  demos_path: reports/demos/demos_{task}.json
```

---

## Notes

- This is a **pure LLM-only** baseline (no hand-crafted Variation Theory rules, no classifier gating).
- Keep the same train/test split for all methods you compare.
- For stability, keep temperature low (0.2–0.3) and outputs short (8–64 tokens).

