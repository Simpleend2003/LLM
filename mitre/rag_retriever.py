# mitre/rag_retriever.py

import re
import torch
from config import TOP_K_EMBED, TOP_K_RERANK
from .knowledge_base import MITREKnowledgeBase


class RAGRetriever:

    def __init__(self, kb: MITREKnowledgeBase):
        self.kb = kb
        self.model = kb.model
        self.tokenizer = kb.tokenizer

    def _heuristic_query_expansion(self, text: str) -> str:
        """
        启发式查询扩展：
        针对过短或缺乏上下文的 Input Text，自动补充战术意图关键词。
        解决向量模型在短文本上“找不准方向”的问题。
        """
        text_lower = text.lower()
        expansion = ""

        # 针对删除/清理行为 -> 补充防御规避语义 (解决 T1070/T1070.004 漏召回)
        if any(w in text_lower for w in ["delete", "remove", "clean", "wipe", "clear"]):
            expansion += " Defense Evasion Artifact Cleanup Indicator Removal"

        # 针对加密/混淆行为 -> 补充混淆语义 (解决 T1001/T1027 漏召回)
        if any(w in text_lower for w in ["encrypt", "encode", "xor", "base64", "obfuscate"]):
            expansion += " Obfuscation Defense Evasion"

        # 针对代理/C2 -> 补充 C2 语义 (解决 T1090/T1071 漏召回)
        if any(w in text_lower for w in ["proxy", "hop", "tunnel", "http", "c2", "beacon"]):
            expansion += " Command and Control Network Communication"

        # 针对凭证/密码 -> 补充凭证获取语义 (解决 T1003 漏召回)
        if any(w in text_lower for w in ["credential", "password", "hash", "dump", "logon"]):
            expansion += " Credential Access Dumping"

        # 如果有扩展词，加在原始文本后面（权重稍低，不破坏原意）
        if expansion:
            return f"{text} [CONTEXT: {expansion}]"
        return text

    def _keyword_force_recall(self, text, current_candidates, top_k_force=10):
        """
        关键词强制召回 (Hard Match)：
        如果文本中直接包含某个 Technique 的 Name 或 ID，无视向量分数，强制将其加入候选列表。
        这是解决 "Credential Dumping" 文本却搜不到 T1003 的最有效手段。
        """
        text_lower = text.lower()
        forced_candidates = []

        # 提取文本中的潜在 ID (如 T1003)
        potential_ids = re.findall(r't\d{4}(?:\.\d{3})?', text_lower)

        for tid, info in self.kb.techniques.items():
            name_lower = info['name'].lower()

            # 规则1：ID 直接匹配
            if tid.lower() in potential_ids:
                forced_candidates.append((tid, 1000.0))  # 给最高分
                continue

            # 规则2：Name 完整包含 (例如 text="performed credential dumping", name="os credential dumping")
            # 这里的逻辑是：如果技术名较长(>3词)且出现在文本里；或者技术名较短但完全匹配
            if len(name_lower) > 4 and name_lower in text_lower:
                forced_candidates.append((tid, 500.0))

            # 规则3：Name 的核心词高度重合 (可选，防止过于宽泛)
            # 这里简单起见，只做精确短语匹配

        # 将强制召回的结果合并到 current_candidates
        # 使用字典去重，保留最高分
        candidate_dict = dict(current_candidates)
        for tid, score in forced_candidates:
            # 强制召回的技术，无论之前有没有，都更新为高分，确保进入 Rerank
            candidate_dict[tid] = max(candidate_dict.get(tid, 0), score)

        # 转回 list 并排序
        merged = list(candidate_dict.items())
        merged.sort(key=lambda x: x[1], reverse=True)

        return merged

    def _keyword_boost(self, text, candidates):
        """
        原来的软性关键词增强 (Soft Boost)
        """
        keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))

        boosted_candidates = []
        for tid, score in candidates:
            info = self.kb.techniques[tid]
            content = (info['name'] + " " + info['description']).lower()
            matches = sum(1 for kw in keywords if kw in content)

            # 稍微降低一点权重，避免干扰强制召回的高分
            new_score = score + (matches * 2)
            boosted_candidates.append((tid, new_score))

        boosted_candidates.sort(key=lambda x: x[1], reverse=True)
        return boosted_candidates

    def retrieve(self, text: str):
        # 0. 预处理：查询扩展
        expanded_query = self._heuristic_query_expansion(text)

        # 1. 向量检索 (Dense Retrieval)
        # 使用扩展后的 Query 进行检索
        inputs = self.kb.tokenizer([expanded_query], padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        q_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]

        # 扩大初筛范围，给后续步骤更多机会
        dense_candidates = self.kb.dense_search(q_emb, top_k=TOP_K_EMBED)

        # 2. 关键词强制召回 (Hard Recall) - 这是修复漏召回的关键步骤
        # 注意：这里用原始 text 匹配，防止扩展词干扰精确匹配
        mixed_candidates = self._keyword_force_recall(text, dense_candidates)

        # 3. 关键词软性增强 (Soft Boost)
        boosted_candidates = self._keyword_boost(text, mixed_candidates)

        # 4. 准备给 Reranker 的数据
        # 此时列表头部是 强制召回(Score=1000) + 向量高分
        # 取前 N 个给精排模型
        top_n_for_rerank = boosted_candidates[:(TOP_K_RERANK * 2)]  # 扩大 Rerank 范围，例如 20*3=60

        # 5. Cross-Encoder 重排 (Reranking)
        # 注意：即使是强制召回的高分，也需要经过 Reranker 确认上下文是否真的相关
        reranked = self.kb.rerank(text, top_n_for_rerank)

        # 6. 最终截断
        final_results = reranked[:TOP_K_RERANK]

        result = []
        for tid, score in final_results:
            info = self.kb.techniques[tid]
            result.append({
                "technique_id": tid,
                "name": info["name"],
                "description": info["description"],
                "tactics": info["tactics"],  # 确保包含 Tactics 供 LLM 判断
                #"score": float(score)
            })

        return result