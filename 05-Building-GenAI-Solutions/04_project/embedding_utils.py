import os
import numpy as np
from typing import Callable, List
from openai import OpenAI


with open('../../../../data/GenAI/openai_key.txt', 'r') as file:
    open_api_key = file.readline().strip()

client = OpenAI(api_key=open_api_key)
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def embed_text_openai(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    v = resp.data[0].embedding
    assert len(v) == 1536
    
    return v

def get_embedder() -> Callable[[str], List[float]]:
    """
    Use environment variable USE_OPENAI=1 to pick OpenAI; otherwise SBERT.
    """
    return embed_text_openai
    