import json
from pathlib import Path
from jinja2 import Template
from ..llm.ollama import OllamaClient

def _render(template_path: str, **kwargs) -> str:
    tpl = Template(Path(template_path).read_text())
    return tpl.render(**kwargs)

def _strict_json(s: str):
    # Handle markdown code blocks
    if "```json" in s:
        start = s.find("```json") + 7
        end = s.find("```", start)
        if end != -1:
            s = s[start:end].strip()
    
    i = s.find("{"); j = s.rfind("}")
    if i == -1 or j == -1 or j <= i: raise ValueError("No JSON in LLM output")
    
    json_str = s[i:j+1]
    try:
        data = json.loads(json_str)
        
        # Handle both old format (single counterfactual) and new format (multiple)
        if "counterfactual" in data:
            # Convert old format to new format
            return {
                "counterfactuals": [
                    {
                        "text": data["counterfactual"],
                        "modification_focus": "general"
                    }
                ]
            }
        elif "counterfactuals" in data:
            return data
        else:
            raise ValueError("Invalid JSON structure")
            
    except json.JSONDecodeError as e:
        # Try to fix common escape sequence issues
        import re
        # Fix common problematic escape sequences
        fixed_json = json_str.replace('\\n', '\\\\n').replace('\\t', '\\\\t').replace('\\r', '\\\\r')
        fixed_json = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', fixed_json)
        try:
            data = json.loads(fixed_json)
            # Apply same format conversion logic
            if "counterfactual" in data:
                return {
                    "counterfactuals": [
                        {
                            "text": data["counterfactual"],
                            "modification_focus": "general"
                        }
                    ]
                }
            elif "counterfactuals" in data:
                return data
            else:
                raise ValueError("Invalid JSON structure")
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON: {e}")

def generate_cf_with_patterns(cfg, task_cfg, original_text: str, from_label: str, to_label: str, candidate_info=None):
    """Generate counterfactual with optional pattern guidance"""
    
    client = OllamaClient(cfg["model_gen"], temperature=cfg["temperature"])
    
    # Use pattern-aware template if candidate info is provided
    if candidate_info and candidate_info.get("strategy") == "pattern_guided":
        template_path = "prompts/generation/gen_counterfactual_with_patterns.txt"
        prompt = _render(template_path,
                        task=cfg.get("task", ""), 
                        text=original_text,
                        from_label=from_label, 
                        to_label=to_label,
                        candidate_phrases=candidate_info.get("phrases", []),
                        pattern_rule=candidate_info.get("pattern_rule", ""))
    else:
        # Fall back to existing template
        prompt = _render(task_cfg["prompts"]["generator"],
                        task=cfg.get("task", ""), 
                        text=original_text,
                        from_label=from_label, 
                        to_label=to_label)
    
    out = client.run(prompt, system=None, max_tokens=cfg["cf_max_new"], retries=1)
    try:
        return _strict_json(out)
    except Exception:
        out = client.run(prompt, system=None, max_tokens=max(128, cfg["cf_max_new"]//2), retries=0)
        return _strict_json(out)

def generate_cf(cfg, task_cfg, original_text: str, from_label: str, to_label: str):
    """Original generate_cf function - unchanged for compatibility"""
    return generate_cf_with_patterns(cfg, task_cfg, original_text, from_label, to_label, None)
