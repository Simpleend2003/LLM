# llm/llm_client.py 改进版
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import GPT_API_KEY, GPT_API_URL, GPT_MODEL

class LLMClient:
    def __init__(self):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def ask(self, prompt: str, temperature=0.1, max_retries=5):
        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4000
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GPT_API_KEY}"
        }

        for i in range(max_retries):
            try:
                resp = self.session.post(
                    GPT_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"第 {i+1} 次请求失败: {e}")
                if i == max_retries - 1:
                    raise
                time.sleep(2 ** i)  # 指数退避
        return None