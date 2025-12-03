# mitre/rag_retriever.py
from config import TOP_K_EMBED, TOP_K_RERANK
from .knowledge_base import MITREKnowledgeBase
import re
import torch

class RAGRetriever:

    def __init__(self, kb: MITREKnowledgeBase):
        self.kb = kb
        self.model = kb.model  # 从 MITREKnowledgeBase 中获取模型
        self.tokenizer = kb.tokenizer  # 从 MITREKnowledgeBase 中获取 tokenizer

    def _keyword_boost(self, text, candidates, boost_score=0.5):
        """
        非常暴力的关键词匹配：
        如果 Input Text 中的关键词出现在 Technique 的名字或描述中，增加其相关性分数。
        """
        # 提取 text 中的有意义关键词 (忽略停用词，这里简单处理，提取长度>3的词)
        keywords = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        C2_KEYWORDS = {"c2", "command and control", "beacon", "https beacon", "smtp", "dns tunnel", "port 443",
                       "alternate protocol"}
        keywords.update(C2_KEYWORDS)
        boosted_candidates = []
        for tid, score in candidates:
            info = self.kb.techniques[tid]
            content = (info['name'] + " " + info['description']).lower()

            # 计算重合词数
            matches = sum(1 for kw in keywords if kw in content)

            # 简单加权：命中一个词加 0.1 分 (根据实际情况调整)
            new_score = score + (matches * 0.001)
            boosted_candidates.append((tid, new_score))

        # 重新排序
        boosted_candidates.sort(key=lambda x: x[1], reverse=True)
        return boosted_candidates

    def retrieve(self, text: str):
        # 1. 向量检索 (Dense Retrieval) - 先取 Top 50，保证范围够大
        inputs = self.kb.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract embeddings (CLS token representation)
        q_emb =outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]

        dense_candidates = self.kb.dense_search(q_emb, top_k=TOP_K_EMBED)

        # 2. 关键词增强 (Keyword Boosting)
        boosted_candidates = self._keyword_boost(text, dense_candidates)

        # 3. 截取前 20 个给 Reranker (省钱/省时间)
        top_n_for_rerank = boosted_candidates[:(TOP_K_RERANK*2)]

        # 4. Cross-Encoder 重排 (Reranking)
        reranked = self.kb.rerank(text, top_n_for_rerank)

        # 5. 最终取 Top 10 返回 (给 LLM 更多选择)
        final_results = reranked[:TOP_K_RERANK]  # 这里一定要给够 10 个！
        #final_results =top_n_for_rerank [:TOP_K_RERANK]  # 这里一定要给够 10 个！
        result = []
        for tid, score in final_results:
            info = self.kb.techniques[tid]
            result.append({
                "technique_id": tid,
                "name": info["name"],
                "description": info["description"],
                "tactics": info["tactics"],
               # "relevance": float(score)
            })

        return result
