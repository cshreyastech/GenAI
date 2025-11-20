"""
Embedding utilities:
- uses OpenAI if available (via openai package)
- otherwise falls back to sentence-transformers (if installed)
- ensures final vector is a plain Python list of floats
"""

import os
from typing import Callable, List
import numpy as np

# Try OpenAI 'openai' package first
USE_OPENAI = bool(os.getenv("USE_OPENAI", "1") == "1")
OPENAI_API_KEY_PATH = os.getenv("OPENAI_KEY_FILE", "../../../../data/GenAI/openai_key.txt")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # adjust as needed

# Lazy imports
_openai = None
_sbert_model = None


def _load_openai():
    global _openai
    if _openai is None:
        try:
            import openai as _openai_pkg
            _openai_pkg.api_key = os.getenv("OPENAI_API_KEY") or _read_api_key_file()
            _openai = _openai_pkg
        except Exception as e:
            _openai = None
    return _openai


def _read_api_key_file() -> str:
    try:
        with open(OPENAI_API_KEY_PATH, "r") as f:
            return f.readline().strip()
    except Exception:
        return ""


def embed_text_openai(text: str) -> List[float]:
    """
    Use OpenAI API (openai package) to create embeddings.
    Returns a Python list of floats.
    """
    client = _load_openai()
    if client is None:
        raise RuntimeError("OpenAI package not available or API key not set")

    resp = client.Embedding.create(input=text, model=OPENAI_EMBED_MODEL)
    emb = resp["data"][0]["embedding"]
    # ensure Python list, not numpy types
    return [float(x) for x in emb]


def embed_text_sbert(text: str) -> List[float]:
    """
    Fallback: create embeddings with sentence-transformers.
    """
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError("sentence-transformers not installed and OpenAI not available") from e
    v = _sbert_model.encode([text])[0]
    return [float(x) for x in v.tolist()]


def get_embedder() -> Callable[[str], List[float]]:
    """
    Returns an embed_fn(text)->List[float], controlled by USE_OPENAI env var.
    """
    if USE_OPENAI:
        return embed_text_openai
    else:
        return embed_text_sbert
