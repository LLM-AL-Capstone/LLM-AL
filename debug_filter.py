#!/usr/bin/env python3

import sys, os
sys.path.append('src')

from src.app.filter.variation_theory_filter import VariationTheoryFilter
from src.app.utils.io import load_yaml

# Load config
cfg = load_yaml("configs/poc.yaml")

# Initialize filter
vt_filter = VariationTheoryFilter(cfg)

# Test with a simple case
original = "i sit outside the hotel room i feel so calm and relaxed"
counterfactual = "I sit inside the hotel room, feeling furious and agitated"
pattern_info = {"strategy": "pattern_guided", "pattern_rule": "Emotional Words and Phrases Pattern"}
target_label = "anger"

print("Testing Combined Filter...")
print(f"Original: {original}")
print(f"Counterfactual: {counterfactual}")
print(f"Target: {target_label}")
print()

# Test the prompt rendering
prompt = vt_filter.combined_template.render(
    original=original,
    counterfactual=counterfactual,
    pattern_rule=pattern_info.get("pattern_rule", ""),
    target_label=target_label
)

print("=== RENDERED PROMPT ===")
print(prompt)
print()

# Get raw LLM response
response = vt_filter.client.run(prompt, system="You are an English-speaking assistant. Respond only in English.", max_tokens=400, retries=1)

print("=== RAW LLM RESPONSE ===")
print(response)
print()

# Parse the response
result = vt_filter._parse_json_response(response)
print("=== PARSED RESULT ===")
print(result)