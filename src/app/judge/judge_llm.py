from jinja2 import Template
from pathlib import Path
import json
from ..llm.ollama import OllamaClient

def judge_llm(cfg, orig_text: str, cf_text: str, target_label: str):
    tpl = Template(Path("prompts/judge.txt").read_text())
    prompt = tpl.render(orig=orig_text, cf=cf_text, target=target_label)
    client = OllamaClient(cfg["model_ann"], temperature=0.2)
    out = client.run(prompt, system=None, max_tokens=cfg.get("judge_max_new", 64))
    
    # Handle markdown code blocks
    if "```json" in out:
        start = out.find("```json") + 7
        end = out.find("```", start)
        if end != -1:
            out = out[start:end].strip()
    
    i, j = out.find("{"), out.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return {"pass_all": False, "score": 0.0, "reasons": {"parse":"fail"}}
    
    json_str = out[i:j+1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common escape sequence issues
        import re
        fixed_json = json_str.replace('\\n', '\\\\n').replace('\\t', '\\\\t').replace('\\r', '\\\\r')
        fixed_json = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', fixed_json)
        try:
            data = json.loads(fixed_json)
        except json.JSONDecodeError:
            return {"pass_all": False, "score": 0.0, "reasons": {"parse":"fail"}}
    overall = 0.25*float(data.get("minimality", 0.0)) + 0.25*float(data.get("fluency", 0.0)) + 0.30*float(data.get("label_determinism", 0.0)) + 0.20*float(data.get("faithfulness", 0.0))
    return {"pass_all": overall >= float(cfg.get("judge_threshold", 0.70)), "score": overall, "reasons": data}
