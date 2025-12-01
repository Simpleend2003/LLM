import json
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from config import EMBEDDING_MODEL, CROSS_ENCODER_MODEL, MITRE_KNOWLEDGE_BASE


class MITREKnowledgeBase:

    def __init__(self):
        self.techniques = {}
        self.tech_ids = []
        self.embeddings = None

        print("Initializing models...")

        # Embedding 模型（你已经换成 MITRE 专用，很好）
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(EMBEDDING_MODEL)
        print("Tokenizers and Models loaded.")

        # ==================== CrossEncoder 加载 + 强制修复 pad_token 问题 ====================
        print(f"Loading CrossEncoder: {CROSS_ENCODER_MODEL}")
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)

        # 关键修复：很多模型（Qwen2/Llama/GPT2）没有 pad_token，导致 batch > 1 报错
        if self.reranker.tokenizer.pad_token is None:
            print("No pad_token found in CrossEncoder tokenizer, setting to eos_token...")
            self.reranker.tokenizer.pad_token = self.reranker.tokenizer.eos_token
            if hasattr(self.reranker.model.config, "pad_token_id"):
                self.reranker.model.config.pad_token_id = self.reranker.tokenizer.eos_token_id
        print("CrossEncoder loaded and pad_token fixed.")
        # ===============================================================================

        print("Loading knowledge base...")
        self._load()
        print("Embedding techniques...")
        self._embed()
        print(f"Techniques loaded: {len(self.techniques)}")
        print(f"Tech IDs loaded: {len(self.tech_ids)}")  # Check tech_ids length here

        print("Models and knowledge base loaded successfully!")

    def _load(self):
        """Loads the MITRE knowledge base from a JSON file."""
        with open(MITRE_KNOWLEDGE_BASE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.techniques = data["techniques"]

    def _embed(self):
        """Embeds the MITRE techniques using the model."""
        embeddings_file = "mitre_embeddings.npy"  # File to save/load embeddings

        # If embeddings are already saved, load them
        if os.path.exists(embeddings_file):
            print("Loading precomputed embeddings...")
            self.embeddings = np.load(embeddings_file)
            self.tech_ids = list(self.techniques.keys())
            return

        # Otherwise, compute embeddings
        corpus = []
        self.tech_ids = []
        for tid, info in self.techniques.items():
            # Prepare the text for encoding
            text = (
                f"Technique ID: {tid}. "
                f"Name: {info['name']}. "
                f"Description: {info['description']}. "
                f"Tactics: {', '.join(info['tactics'])}. "
            )
            corpus.append(text)
            self.tech_ids.append(tid)

        print(f"Encoding {len(corpus)} MITRE techniques...")

        # Create inputs in small batches to avoid memory overload
        batch_size = 8  # You can adjust this based on available memory
        all_embeddings = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i + batch_size]

            # Tokenize the batch (Ensure proper padding and truncation)
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)

            # Move inputs to the same device as the model (if using GPU, or leave on CPU)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            # Forward pass: Get model output
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Extract embeddings from the last hidden state (usually from the [CLS] token)
            embeddings = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()

            # Append embeddings to the list
            all_embeddings.extend(embeddings)

        # Convert the list of embeddings to a numpy array
        self.embeddings = np.array(all_embeddings)
        print(f"Encoding complete. {len(self.embeddings)} embeddings created.")

        # Save embeddings to a file
        np.save(embeddings_file, self.embeddings)
        print(f"Embeddings saved to {embeddings_file}.")

    def dense_search(self, query_emb, top_k=20):
        """Searches the embeddings for the top_k most relevant techniques."""
        scores = np.dot(self.embeddings, query_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.tech_ids[i], float(scores[i])) for i in top_idx]

    def rerank(self, query, candidates):
        """Re-ranks the candidate techniques based on the query."""
        texts = []
        for tid, _ in candidates:
            info = self.techniques[tid]
            texts.append([query, info["description"]])

        # Predict using the cross-encoder model
        scores = self.reranker.predict(texts)
        reranked = list(zip([c[0] for c in candidates], scores))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
