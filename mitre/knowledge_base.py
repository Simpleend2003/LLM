import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from config import EMBEDDING_MODEL, CROSS_ENCODER_MODEL, MITRE_KNOWLEDGE_BASE


class MITREKnowledgeBase:

    def __init__(self):
        self.techniques = {}
        self.tech_ids = []
        self.embeddings = None

        # Initialize the models only once
        print("Initializing models...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)  # Only once during initialization
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)  # Only once during initialization

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
        """Embeds the MITRE techniques using the SentenceTransformer."""
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
               # f"Procedure Examples: {', '.join(info.get('examples', []))}"
            )
            corpus.append(text)
            self.tech_ids.append(tid)

        print(f"Encoding {len(corpus)} MITRE techniques...")
        # Ensure we only call encoder.encode once
        self.embeddings = self.encoder.encode(corpus, normalize_embeddings=True)
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
