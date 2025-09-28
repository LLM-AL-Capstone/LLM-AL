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
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common escape sequence issues
        import re
        # Fix common problematic escape sequences
        fixed_json = json_str.replace('\\n', '\\\\n').replace('\\t', '\\\\t').replace('\\r', '\\\\r')
        fixed_json = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', fixed_json)
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON: {e}")

def generate_cf(cfg, task_cfg, original_text: str, from_label: str, to_label: str):
    client = OllamaClient(cfg["model_gen"], temperature=cfg["temperature"])
    prompt = _render(task_cfg["prompts"]["generator"],
                     task=cfg["task"], text=original_text,
                     from_label=from_label, to_label=to_label)
    out = client.run(prompt, system=None, max_tokens=cfg["cf_max_new"], retries=1)
    try:
        return _strict_json(out)
    except Exception:
        out = client.run(prompt, system=None, max_tokens=max(128, cfg["cf_max_new"]//2), retries=0)
        return _strict_json(out)
