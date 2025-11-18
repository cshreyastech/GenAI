import os
from real_estate_db import RealEstateDBManager
from embedding_utils import get_embedder #, USE_OPENAI
from rag_pipeline import RAGEngine

DB_PATH = os.getenv("LANCEDB_PATH", "../../../../data/GenAI/05_project/lancedb_store")
json_path = "listings.json"

def main():
    json_path = "listings.json"
    manager = RealEstateDBManager(DB_PATH)
    manager.create_table(force=True)

    embed_fn = get_embedder()
    
    # Computes embeddings and stores in LanceDB
    manager.ingest_listings(json_path, embed_fn)

    # optional: build index if supported
    # manager.create_vector_index(index_type="hnsw", metric="cosine")

    # RAG engine
    rag = RAGEngine(manager, embed_fn)

    # Query
    user_query = "Looking for a modern 3-bedroom with good schools, near public transit, under $1.5M"
    result = rag.query(user_query, k=5)
    print("Result:")
    print(result)

if __name__ == "__main__":
    main()