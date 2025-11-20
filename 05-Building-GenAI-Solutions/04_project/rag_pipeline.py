"""
RAGEngine - given a RealEstateDBManager and an embed function, run retrieval and call chat LLM.
Outputs JSON-serializable dicts (safe for json.dumps).
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import json
import openai  # using standard openai package
from real_estate_db import RealEstateDBManager

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not OPENAI_API_KEY:
    # try file fallback
    try:
        with open(os.getenv("OPENAI_KEY_FILE", "../../../../data/GenAI/openai_key.txt"), "r") as f:
            OPENAI_API_KEY = f.readline().strip()
    except Exception:
        OPENAI_API_KEY = None

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas dtypes to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _cosine_similarity_matrix(emb_matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of emb_matrix and q (both numpy arrays)."""
    emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    sims = (emb_norm @ q_norm).astype(float)
    return sims


class RAGEngine:
    def __init__(self, db_manager: RealEstateDBManager, embed_fn):
        self.db = db_manager
        self.embed_fn = embed_fn

    def query(self, user_query: str, k: int = 5) -> Dict[str, Any]:
        qv = np.array(self.embed_fn(user_query), dtype=float)

        # 1) try native LanceDB search
        lres = None
        try:
            lres = self.db.search_with_lancedb(qv.tolist(), limit=k)
        except Exception as e:
            lres = None

        contexts = []
        if lres is not None and not lres.empty:
            # convert pandas rows into safe dicts
            for _, row in lres.iterrows():
                contexts.append({
                    "id": str(row["id"]),
                    "score": float(row.get("score", 0.0)) if "score" in row else None,
                    "full_text": str(row["full_text"]),
                    "neighborhood": str(row.get("neighborhood", "")),
                    "price": str(row.get("price", "")),
                    "bedrooms": int(row.get("bedrooms", 0)) if row.get("bedrooms", None) is not None else None,
                    "bathrooms": int(row.get("bathrooms", 0)) if row.get("bathrooms", None) is not None else None,
                })
            return self._generate_answer(user_query, qv, contexts)

        # 2) fallback: load all embeddings, compute cosine similarities in-memory
        ids, embeddings, df = self.db.fetch_all_embeddings()
        if len(embeddings) == 0:
            return {"query": user_query, "top_k": []}

        emb_arr = np.array(embeddings, dtype=float)
        sims = _cosine_similarity_matrix(emb_arr, qv)
        top_idx = np.argsort(-sims)[:k]

        for idx in top_idx:
            row = df.iloc[idx]
            contexts.append({
                "id": str(row["id"]),
                "score": float(sims[idx]),
                "full_text": str(row["full_text"]),
                "neighborhood": str(row.get("neighborhood", "")),
                "price": str(row.get("price", "")),
                "bedrooms": int(row.get("bedrooms", 0)) if row.get("bedrooms", None) is not None else None,
                "bathrooms": int(row.get("bathrooms", 0)) if row.get("bathrooms", None) is not None else None,
            })

        return self._generate_answer(user_query, qv, contexts)

    def _generate_answer(self, user_query: str, qv, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a prompt and call the LLM. If OpenAI is not configured, return the contexts
        as a JSON-safe debug response.
        """
        # Always sanitize contexts
        contexts = _to_json_safe(contexts)

        if not openai.api_key:
            return {"query": user_query, "top_k": contexts}

        system_prompt = (
            "You are an assistant that recommends real estate listings based on a user's natural-language requirements. "
            "Given the query and the retrieved candidate listings below, produce a short ranked recommendation with reasoning and actionable next steps."
        )

        context_texts = []
        for c in contexts:
            # keep listing short so LLM prompt remains compact
            context_texts.append(f"Listing ID {c['id']} (score={c['score']:.4f}): {c['full_text']}")

        user_prompt = (
            f"User query: {user_query}\n\n"
            "Retrieved candidate listings (top results):\n" +
            "\n\n".join(context_texts) +
            "\n\nTask: Recommend the top 3 listings, explain why each matches the user's intent, and list missing info or next steps."
        )

        # call OpenAI ChatCompletion
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            n=1
        )

        answer_text = resp["choices"][0]["message"]["content"]
        result = {"answer": answer_text, "retrieved": contexts}
        return _to_json_safe(result)
