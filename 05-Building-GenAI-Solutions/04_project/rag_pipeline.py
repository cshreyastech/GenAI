import os
import numpy as np
from typing import List

from real_estate_db import RealEstateDBManager
from embedding_utils import get_embedder, client

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    return float(np.dot(a, b))

class RAGEngine:
    def __init__(self, db_manager: RealEstateDBManager, embed_fn):
        self.db = db_manager
        self.embed_fn = embed_fn

    def query(self, user_query: str, k: int = 5):
        # 1) run the query
        qv = np.array(self.embed_fn(user_query), dtype=float)

        # 2) try native LanceDB search
        lres = self.db.search_with_lancedb(qv.tolist(), limit=k)
        if lres is not None and not lres.empty:
            # Assume lres is a pandas DataFrame: pick top k rows
            contexts = []
            for _, row in lres.iterrows():
                contexts.append({
                    "id": int(row["id"]),
                    "score": float(row.get("score", 0.0)) if "score" in row else None,
                    "full_text": row["full_text"],
                    **{c: row[c] for c in ["neighborhood","price","bedrooms","bathrooms"] if c in row}
                })
            return self._generate_answer(user_query, qv, contexts)
        # 3) fallback: load all embeddings and compute cosine in-memory
        ids, embeddings, df = self.db.fetch_all_embeddings() # 10, 10, (10, 10)
        emb_arr = np.array(embeddings, dtype=float)

        # Normalize and compute similarity
        emb_norm = emb_arr / (np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-12)
        q_norm = qv / (np.linalg.norm(qv) + 1e-12)

        sims = (emb_norm @ q_norm).astype(float) # cosine similarity
        top_idx = np.argsort(-sims)[:k]

        contexts = []
        for idx in top_idx:
            row = df.iloc[idx]
            contexts.append({
                # "id": int(row["id"]),
                "id": row["id"],
                "score": float(sims[idx]),
                "full_text": row["full_text"],
                "neighborhood": row.get("neighborhood", ""),
                "price": row.get("price", ""),
                "bedrooms": row.get("bedrooms", None),
                "bathrooms": row.get("bathrooms", None),
            })
        return self._generate_answer(user_query, qv, contexts)
        
    def _generate_answer(self, user_query: str, qv, contexts: List[dict]):
            """
            Compose a RAG prompt and call LLM (OpenAI path shown).
            If no OpenAI, simply return the contexts and scores.
            """
            if client is None:
                # return contexts as a lightweight answer for local debug
                return {
                    "query": user_query,
                    "top_k": contexts,
                }
    
            # Build system + user messages
            system_prompt = (
                "You are an assistant that recommends real estate listings based on a user's natural-language requirements. "
                "Given the query and the retrieved candidate listings below, produce a short ranked recommendation with reasoning and actionable next steps."
            )
    
            # assemble context block
            context_texts = []
            for c in contexts:
                context_texts.append(f"Listing ID {c['id']} (score={c['score']:.4f}): {c['full_text']}")
    
            user_prompt = (
                f"User query: {user_query}\n\n"
                "Retrieved candidate listings (top results):\n" +
                "\n\n".join(context_texts) +
                "\n\nTask: Recommend the top 3 listings, explain why each matches the user's intent, and list missing info or next steps."
            )
    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
    
            resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=600)
            text = resp.choices[0].message.content
            return {"answer": text, "retrieved": contexts}