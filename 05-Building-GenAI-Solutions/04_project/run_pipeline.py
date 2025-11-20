"""
Entry point to run the ingestion + RAG query demo.

Before running:
- Ensure OPENAI_API_KEY is set in the environment or OPENAI_KEY_FILE path exists.
- Ensure LANCE DB path points to a writeable directory.
- Listings file used in this demo: /mnt/data/listings.json
"""

import os
from real_estate_db import RealEstateDBManager
from embedding_utils import get_embedder
from rag_pipeline import RAGEngine

DB_PATH = os.getenv("LANCEDB_PATH", "../../../../data/GenAI/05_project/lancedb_store")
JSON_PATH = os.getenv("LISTINGS_JSON", "listings.json")  # Listings file

def main():
    manager = RealEstateDBManager(DB_PATH)
    # create_table(force=True) will drop old table.
    manager.create_table(force=True)

    embed_fn = get_embedder()

    # Ingest (deduplicated)
    manager.ingest_listings(JSON_PATH, embed_fn)

    # Try to create index (best-effort)
    manager.create_vector_index(index_type="hnsw", metric="cosine")

    # Run a sample RAG query
    rag = RAGEngine(manager, embed_fn)
    user_query = "Looking for a modern 3-bedroom with good schools, near public transit, under $1.5M"
    result = rag.query(user_query, k=5)

    # Print JSON-safe output
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()