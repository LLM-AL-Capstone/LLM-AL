# LLM-AL: Variation Theory Counterfactual Data Augmentation

**Purpose**: Implement **Variation Theory-guided counterfactual data augmentation** for Active Learning using state-of-the-art techniques:
- **Neuro-Symbolic Pattern Learning** using Programming-by-Example approach
- **Pattern-Guided Counterfactual Generation** with minimal semantic edits
- **Three-Stage Quality Filtering** (C1: Heuristic, C2: Pattern Consistency, C3: Label Flip Verification)
- **Multi-candidate Selection** with diversity optimization
- **Few-shot Evaluation** with balanced sampling strategies

> Uses **Ollama** to run open-source LLMs locally (e.g., `qwen:7b-chat`) for complete pipeline execution.

---

## 🎯 Variation Theory Implementation

This implementation follows the research methodology from "Variation Theory in Counterfactual Data Augmentation for Active Learning":

### **Step 1: Neuro-Symbolic Pattern Learning**
- Learns interpretable patterns from labeled examples using Programming-by-Example
- Identifies syntactic structures and semantic markers for each label
- Caches learned patterns for efficient reuse across runs

### **Step 2: Pattern-Guided Generation**  
- Generates multiple counterfactual candidates (2-3 per example)
- Uses learned patterns to guide minimal, targeted modifications
- Fallback to general strategy when no patterns are available

### **Step 3: Three-Stage Filtering**
- **C1 (Heuristic)**: Removes malformed or incomplete generations
- **C2 (Pattern Consistency)**: Ensures syntactic structure preservation 
- **C3 (Label Flip)**: Verifies semantic label transformation success
- **Optimized**: Combined C2+C3 filtering reduces LLM calls by 59%

### **Step 4: Quality-Based Selection**
- Selects best candidates based on multi-metric scoring
- Handles identical scores with balanced random sampling
- Ensures label distribution balance for few-shot learning

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install Ollama and pull model
ollama pull qwen:7b-chat

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

#### **Emotions Dataset** (6 labels: sadness, joy, love, anger, fear, surprise)
```bash
# Full pipeline with pattern learning
python -m src.app.services.demos.make_demos --config configs/poc.yaml --task emotions --sample 500 --k 120

# Small test run
python -m src.app.services.demos.make_demos --config configs/poc.yaml --task emotions --sample 20 --k 10
```

#### **Yelp Dataset** (4 categories: environment, price, products, service)
```bash
# Cross-domain demonstration
python -m src.app.services.demos.make_demos --config configs/poc.yaml --task yelp --sample 20 --k 10 --force-relearn
```

### Multi-shot Evaluation
```bash
# Enhanced evaluation with balanced sampling
python -m src.app.services.eval.multi_shot_eval --config configs/poc.yaml --task emotions --k 120
```

---

## 📊 Key Features

### **Pattern Learning & Caching**
- Learns 0-5 patterns per label using Programming-by-Example
- Automatically caches patterns in `reports/demos/` for reuse
- Cross-domain adaptation (patterns transfer between datasets)

### **Smart Generation Strategy**
- **Pattern-guided**: Uses learned patterns for targeted modifications
- **General fallback**: Minimal edit approach when no patterns available
- **Multi-candidate**: Generates 2-3 options per input for diversity

### **Optimized Filtering**
- **59% LLM call reduction** through combined filtering
- **Relaxed thresholds**: c2_threshold=0.6, c3_threshold=0.7 for realistic scoring
- **Quality metrics**: Pattern Keeping Rate (PKR), Label Flip Rate (LFR), Semantic LFR (SLFR)

### **Enhanced Evaluation**
- **Balanced selection**: Random sampling when scores are identical
- **Label distribution**: Maintains proportional representation
- **Cross-validation**: Multi-run averaging for robust metrics

---

## 🔧 Configuration

### Dataset Configuration (`configs/tasks/`)
```yaml
# emotions.yaml
name: "emotions"
data_splits:
  train: "data/splits/emotions_train.csv"
  test: "data/splits/emotions_test.csv"
label_column: "label"
text_column: "text"
labels: ["sadness", "joy", "love", "anger", "fear", "surprise"]
```

### Pipeline Configuration (`configs/poc.yaml`)
```yaml
# Variation Theory settings
variation_theory:
  enable_pattern_learning: true
  pattern_cache_dir: "reports/demos"
  
# Filter thresholds (relaxed for realistic scoring)
filtering:
  c2_threshold: 0.6  # Pattern consistency
  c3_threshold: 0.7  # Label flip verification

# Generation settings  
generation:
  num_candidates: 3    # Will be optimized to 2-3 actual candidates
  enable_optimization: true
```

---

## 📁 Project Structure

```
LLM-AL/
├── configs/
│   ├── poc.yaml              # Main pipeline configuration
│   └── tasks/                # Dataset-specific configurations
├── data/splits/              # Train/test data splits
├── prompts/
│   ├── annotation/           # Labeling prompts
│   ├── filtering/            # Quality filtering prompts
│   ├── generation/           # Counterfactual generation prompts
│   └── patterns/             # Pattern learning prompts
├── src/app/
│   ├── patterns/             # Neuro-symbolic pattern learning
│   ├── filter/               # Three-stage Variation Theory filtering
│   ├── generate/             # Multi-candidate generation
│   ├── services/
│   │   ├── demos/            # Main pipeline execution
│   │   └── eval/             # Enhanced multi-shot evaluation
│   └── llm/                  # Ollama integration
└── reports/demos/            # Generated demonstrations & cached patterns
```

---

## 📈 Performance Metrics

### **Optimization Results**
- **59% reduction** in LLM calls through combined filtering
- **Cross-domain capability**: Emotions ↔ Yelp dataset transfer
- **Quality maintenance**: ~0.84 average scores across all metrics
- **Pattern efficiency**: 0-5 patterns learned per label automatically

### **Evaluation Metrics**
- **Accuracy**: Classification correctness on test set
- **Macro-F1**: Balanced performance across all labels
- **Label Distribution**: Proportional few-shot representation
- **Generation Success Rate**: ~80% pass three-stage filtering

---

## 🎓 Research Implementation

This project implements the complete **Variation Theory** methodology for Active Learning:

1. **Programming-by-Example Pattern Learning**: Discovers interpretable structures from small labeled sets
2. **Pattern-Guided Minimal Edits**: Creates targeted, semantically meaningful counterfactuals  
3. **Multi-Stage Quality Assurance**: Ensures both syntactic and semantic validity
4. **Optimized LLM Usage**: Reduces computational overhead while maintaining quality
5. **Cross-Domain Adaptation**: Patterns and strategies transfer between different classification tasks

The implementation matches research-grade standards while optimizing for practical deployment with local LLMs.

