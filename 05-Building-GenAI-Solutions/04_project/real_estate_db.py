import os
import json
from typing import List, Optional

import lancedb
from lancedb.pydantic import LanceModel, vector

# Adjust embedding dim at runtime if needed
DEFAULT_EMBED_DIM = int(os.getenv("EMBED_DIM", 256)) #1536

class RealEstateListing(LanceModel):
    id: int
    neighborhood: str
    price: str
    bedrooms: float
    bathrooms: float
    house_size: str
    description: str
    neighborhood_description: str
    full_text: str
    embedding: vector(1536)

class RealEstateDBManager:
    def __init__(self, db_path: str, table_name: str = "real_estate_listing"):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None

    def create_table(self, force: bool = True):
        if force:
            self.db.drop_table(self.table_name, ignore_missing=True)
        self.table = self.db.create_table(self.table_name, schema=RealEstateListing)
        print(f"Created LanceDB table: {self.table_name}")

    def make_full_text(self, raw: dict) -> str:
        # Compose a human-readable unified text for embeddings
        parts = [
            f"Neighborhood: {raw.get('neighborhood','')}",
            (f"Area: {raw.get('neighborhood_description','')}") if raw.get('neighborhood_description') else "",
            f"Price: {raw.get('price','')}",
            f"Bedrooms: {raw.get('bedrooms','')}, Bathrooms: {raw.get('bathrooms','')}",
            f"Size: {raw.get('house_size','')}",
            f"Details: {raw.get('description','')}"
        ]
        return ". ".join([p for p in parts if p]).strip()

    def ingest_listings(self, json_path: str, embed_fn):
        """
        Compute embedding via embed_fn(full_text) -> list[float]
        embed_fn is a callable that returns a list/ndarray (embedding vector)
        """

        with open(json_path, 'r') as file:
            data = json.load(file)


        docs = []
        for i, listing in enumerate(data["listings"]):
            full_text = self.make_full_text(listing)
            emb = embed_fn(full_text)
            
            if emb is None or len(emb) != 1536:
                raise RuntimeError(f"Invalid embedding dimension: {len(emb)}")
        
            item = {
                "id": i,
                "neighborhood": listing.get("neighborhood", ""),
                "price": listing.get("price", ""),
                "bedrooms": listing.get("bedrooms", 0),
                "bathrooms": listing.get("bathrooms", 0),
                "house_size": listing.get("house_size", ""),
                "description": listing.get("description", ""),
                "neighborhood_description": listing.get("neighborhood_description", ""),
                "full_text": full_text,
                "embedding": emb,
            }
            docs.append(item)

        # create table if missing
        if self.table is None:
            self.create_table(force=False)  # do not drop by default
        # add documents
        self.table.add(docs)
        print(f"Ingested {len(docs)} listings")

    def create_vector_index(self, index_type: str = "hnsw", metric: str = "cosine", **kwargs):
        """
        Optional: Create a vector index
        """
        try:
            self.table.create_index("embedding", index_type=index_type, metric=metric, **kwargs)
            print("Created vector index on embedding")
        except Exception as e:
            print("Could not create index (maybe your LanceDB version lacks this API).", e)