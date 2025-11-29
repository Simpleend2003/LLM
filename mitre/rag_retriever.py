# mitre/rag_retriever.py

from sentence_transformers import SentenceTransformer
from config import TOP_K_EMBED, TOP_K_RERANK
from .knowledge_base import MITREKnowledgeBase


class RAGRetriever:

    def __init__(self, kb: MITREKnowledgeBase):
        self.kb = kb
        self.encoder = kb.encoder

    def retrieve(self, text: str):
        q_emb = self.encoder.encode([text], normalize_embeddings=True)[0]

        # Step 1: dense retrieval
        dense = self.kb.dense_search(q_emb, top_k=TOP_K_EMBED)

        # Step 2: reranking
        reranked = self.kb.rerank(text, dense)
        reranked = reranked[:TOP_K_RERANK]

        result = []
        for tid, score in reranked:
            info = self.kb.techniques[tid]
            result.append({
                "technique_id": tid,
                "name": info["name"],
                "description": info["description"],
                "tactics": info["tactics"],
                "relevance": float(score)
            })

        return result
