# llm/llm_client.py

import requests
from config import GPT_API_KEY, GPT_API_URL, GPT_MODEL


class LLMClient:

    def ask(self, prompt: str, temperature=0.1):
        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2000
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GPT_API_KEY}"
        }

        resp = requests.post(GPT_API_URL, json=payload, headers=headers)
        resp.raise_for_status()

        return resp.json()["choices"][0]["message"]["content"]
