# config.py

MITRE_KNOWLEDGE_BASE = "data/mitre_attack_knowledge_base.json"

#EMBEDDING_MODEL = "cset/sent-distilroberta-cset-base"
EMBEDDING_MODEL = "sarahwei/MITRE-v15-tactic-bert-case-based"      # 专为 ATT&CK 训练
#sarahwei/mitre-attack-cross-encoder
#CROSS_ENCODER_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"  # 开源最
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GPT_API_URL = "https://api.vveai.com/v1/chat/completions"
GPT_MODEL = "gpt-4.1-mini"
GPT_API_KEY = "sk-QxY2t5FPmHe2TF6xEcBaE3B1BcE24b1194B2Fb9422543270"
# config.py (建议值)
TOP_K_EMBED = 100   # 粗排多召回一些
TOP_K_RERANK = 10  # 精排给 LLM 10 个
