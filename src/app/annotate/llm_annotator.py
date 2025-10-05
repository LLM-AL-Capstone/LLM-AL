import json
from pathlib import Path
from jinja2 import Template
from ..llm.ollama import OllamaClient

def _render(path: str, **kwargs):
    return Template(Path(path).read_text()).render(**kwargs)

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

def annotate_label(cfg, task_cfg, text: str, labels):
    client = OllamaClient(cfg["model_ann"], temperature=0.2)
    tpl = Template(Path("prompts/annotation/annotator.txt").read_text())
    prompt = tpl.render(text=text, labels=", ".join(labels))
    out = client.run(prompt, system=None, max_tokens=cfg["ann_max_new"], retries=1)
    try:
        return _strict_json(out)
    except Exception:
        try:
            out = client.run(prompt, system=None, max_tokens=max(16, cfg["ann_max_new"]), retries=0)
            return _strict_json(out)
        except Exception:
            # Fallback: return first label if JSON parsing completely fails
            return {"label": labels[0]}
