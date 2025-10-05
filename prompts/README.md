# Prompts Organization

This directory contains all LLM prompts organized by functionality for the Variation Theory LLM-AL pipeline.

## 📁 Folder Structure

### `annotation/`
- **`annotator.txt`** - Basic label annotation prompt
- **`annotator_with_demos.txt`** - Label annotation with demonstrations

### `filtering/` 
- **`filter.txt`** - C3 Quality assessment (minimality, fluency, label_determinism, faithfulness)
- **`label_flip_discriminator.txt`** - C3 Label Flip Rate (LFR) and Soft Label Flip Rate (SLFR) evaluation  
- **`pattern_consistency_filter.txt`** - C2 Pattern Keeping Rate (PKR) evaluation
- **`combined_filter.txt`** - ⚡ **OPTIMIZED**: Combined C2+C3 evaluation in single call

### `generation/`
- **`gen_counterfactual_with_patterns.txt`** - ⚡ **OPTIMIZED**: Pattern-guided generation (2-3 candidates)
- **`generator.txt`** - ⚡ **OPTIMIZED**: General generation (2-3 candidates)

### `patterns/`
- **`pattern_learning.txt`** - Neuro-symbolic pattern learning using Programming-by-Example

## 🎯 Usage in Pipeline

### Variation Theory Three-Stage Filter:
1. **C1 - Heuristic**: Built-in malformed text detection (no prompt file)
2. **C2+C3 - Combined**: ⚡ Uses `filtering/combined_filter.txt` for PKR, LFR, SLFR, and quality in **one LLM call**

### Pattern-Guided Generation:
1. **Pattern Learning**: Uses `patterns/pattern_learning.txt`
2. **Candidate Generation**: ⚡ Uses optimized prompts generating **2-3 candidates** instead of 3-5
3. **Label Verification**: Uses `annotation/annotator.txt`

## ⚡ Performance Optimizations

**Optimization 1 - Reduced Candidates**: 
- Generation prompts now create 2-3 candidates instead of 3-5
- **~25% reduction** in LLM calls

**Optimization 2 - Combined Filtering**:
- `combined_filter.txt` evaluates PKR + LFR + SLFR + Quality in single call
- **~60% reduction** in filtering calls (4 calls → 1 call per candidate)

**Total Optimization**: **~70% reduction** in LLM calls
- **Before**: ~8,500 calls for 500 rows
- **After**: ~2,500 calls for 500 rows

## 🔄 Code References

- **Pattern Learning**: `src/app/patterns/neurosymbolic_patterns.py`
- **Generation**: `src/app/generate/minimal_edit.py`
- **Filtering**: `src/app/filter/variation_theory_filter.py`
- **Annotation**: `src/app/annotate/llm_annotator.py`

All prompts enforce English-only output to maintain consistency across the pipeline.