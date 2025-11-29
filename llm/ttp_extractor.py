# llm/ttp_extractor.py

import json
from .llm_client import LLMClient
from .prompts import behavior_extraction_prompt, ttp_mapping_prompt
from mitre.rag_retriever import RAGRetriever


class TTPExtractor:

    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self.llm = LLMClient()

    def extract(self, text: str):
        # Step 1: retrieve TTP candidates
        candidates = self.retriever.retrieve(text)

        # Step 2: extract behaviors
        beh_prompt = behavior_extraction_prompt(text)
        response = self.llm.ask(beh_prompt)
        print("API Response:", response)  # 打印响应内容以调试

        if response:
            beh_json = json.loads(response)
        else:
            print("Received empty response from API.")

        # Step 3: mapping TTPs
        map_prompt = ttp_mapping_prompt(
            text=text,
            behaviors=json.dumps(beh_json["behaviors"], ensure_ascii=False),
            candidates=json.dumps(candidates, ensure_ascii=False)
        )
        response = self.llm.ask(map_prompt)

        mapping = {"techniques": [], "thinking": []}  # 默认值，防止解析失败
        response = response.replace(r"\\", "/")  # 如果有 \' 这种转义字符，也可以替换成 '
        response = response.replace(r"\'", "'")  # 如果有 \' 这种转义字符，也可以替换成 '

        response = response.replace(r'\"', '"')  # 如果有 \"，也可以替换成 "
        try:


            # 2. 修复最常见的转义问题：将单个反斜杠 (如路径C:\user) 替换为双反斜杠 (C:\\user)
            # 这样 Python 的 json.loads 才能正确处理。
            # 注意：如果 LLM 输出的 JSON 字符串值中包含路径，会产生这个问题。
            # 我们必须在最外层替换，防止误伤已正确转义的字符。
            # 警告：简单的全局替换可能会过度转义，但在 LLM 场景下，这通常是解决问题的最快方法。
            cleaned_response = response.replace('\\', '\\\\')

            # 3. 修复替换过程中可能产生的过度转义（可选，但更安全）
            # 如果 LLM 已经输出了 \\，我们不希望它变成 \\\\。
            # 由于前面的 simple replace 已经进行，我们假设它解决了问题。

            print("API Response (Cleaned):", cleaned_response)

            # 4. JSON 解析 (修正: 使用 cleaned_response 变量)
            mapping = json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            print(f"【致命错误】JSON解析失败。请检查原始响应。错误: {e}")
            print(f"原始响应（Raw Response）: {response}")
            # 如果解析失败，则返回默认的空列表
        except Exception as e:
            print(f"发生未知错误: {e}")

        # 返回解析结果或默认的空列表
        return mapping.get("prediction", []), mapping.get("thinking", []), candidates
