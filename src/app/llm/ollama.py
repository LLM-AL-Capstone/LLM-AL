import requests, os, time
from typing import Optional

class OllamaClient:
    def __init__(self, model: str, host: Optional[str] = None, temperature: float = 0.25):
        self.model = model
        self.temperature = temperature
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.url = f"{self.host}/api/generate"

    def run(self, prompt: str, system: Optional[str] = None, max_tokens: int = 64, retries: int = 1, timeout: int = 600) -> str:
        full_prompt = prompt if not system else f"<|system|>\n{system}\n<|user|>\n{prompt}"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
            "stream": False,
        }
        for attempt in range(retries + 1):
            try:
                r = requests.post(self.url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception:
                if attempt >= retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
