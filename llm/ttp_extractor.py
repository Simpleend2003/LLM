# llm/ttp_extractor.py

import json
from .llm_client import LLMClient
from .prompts import ttp_mapping_cot_prompt
from mitre.rag_retriever import RAGRetriever


class TTPExtractor:

    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.llm = LLMClient()

    def extract(self, text: str):
        # Step 1: Retrieve - 增加 Top K 到 10，防止漏召回
        # 注意：这需要 config.py 中的 TOP_K_RERANK 至少为 10，否则这里取不到 10 个
        candidates_raw = self.retriever.retrieve(text)[:10]

        # Step 2: Format with Ranking
        candidates_str = ""
        for i, c in enumerate(candidates_raw):
            rank = i + 1
            candidates_str += f"--- [Rank {rank}] ---\n"
            candidates_str += f"ID: {c['technique_id']}\n"
            candidates_str += f"Name: {c['name']}\n"
            # 描述保持压缩，防止 10 个候选项撑爆 Prompt 上下文
            desc = c['description'][:150].replace("\n", " ") + "..."
            candidates_str += f"Description: {desc}\n\n"

        # Step 3: Call LLM
        map_prompt = ttp_mapping_cot_prompt(
            text=text,
            candidates=candidates_str
        )

        response = self.llm.ask(map_prompt)

        # --- JSON 清洗与解析 ---
        mapping = {"prediction": [], "analysis": ""}
        try:
            cleaned_response = response.strip()

            # 清理 Markdown 代码块
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:].strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()

            # 强化清洗
            cleaned_response = cleaned_response.replace('\n', ' ').replace('\r', ' ')
            cleaned_response = cleaned_response.replace('\\', '\\\\')

            data = json.loads(cleaned_response)

            # 确保 prediction 字段是列表
            mapping["prediction"] = data.get("prediction", [])
            if not isinstance(mapping["prediction"], list):
                mapping["prediction"] = [mapping["prediction"]]

            mapping["analysis"] = data.get("analysis", "No analysis")

        except Exception as e:
            print(f"【JSON解析失败】: {e}")
            print(f"原始响应: {response[:200]}...")
            # 兜底策略：如果解析失败，尝试直接使用 Rank 1 的结果
            if candidates_raw:
                print("Using Rank 1 candidate as fallback.")
                mapping["prediction"] = [candidates_raw[0]['technique_id']]

        return mapping["prediction"], [mapping["analysis"]], candidates_raw