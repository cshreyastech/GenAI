import os
import json
from typing import List, Optional

import lancedb
from lancedb.pydantic import LanceModel, vector
import hashlib

# DEFAULT_EMBED_DIM should be the embedding length of the model
DEFAULT_EMBED_DIM = int(os.getenv("EMBED_DIM", 1536)) 




def compute_listing_id(full_text: str):
    return hashlib.md5(full_text.encode()).hexdigest()

class RealEstateListing(LanceModel):
    id: str  # <-- IMPORTANT: now a string!
    neighborhood: str
    price: str
    bedrooms: float
    bathrooms: float
    house_size: str
    description: str
    neighborhood_description: str
    full_text: str
    embedding: List[float]  # <-- VECTOR FIELD

class RealEstateDBManager:
    def __init__(self, db_path: str, table_name: str = "real_estate_listing"):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None

    def create_table(self, force: bool = False):
        """Create or load the LanceDB table."""
        table_names = self.db.table_names()
    
        # If table exists and we do NOT want to force recreate → open it
        if self.table_name in table_names and not force:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing LanceDB table: {self.table_name}")
            return
    
        # Otherwise recreate the table
        if force:
            self.db.drop_table(self.table_name, ignore_missing=True)
            print(f"Dropped old table: {self.table_name}")
    
        # Create new table
        self.table = self.db.create_table(
            self.table_name,
            schema=RealEstateListing
        )
        print(f"Created new LanceDB table: {self.table_name}")

    def get_or_create_table(self):
        if self.table is not None:
            return self.table
    
        table_names = self.db.table_names()
    
        if self.table_name in table_names:
            self.table = self.db.open_table(self.table_name)
            print(f"Loaded existing table: {self.table_name}")
        else:
            self.table = self.db.create_table(self.table_name, schema=RealEstateListing)
            print(f"Created new table: {self.table_name}")
    
        return self.table

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
        table = self.get_or_create_table()
    
        # Load existing IDs
        existing_ids = set(row["id"] for row in table.to_pandas()[["id"]].to_dict("records"))
    
        with open(json_path, "r") as f:
            data = json.load(f)
    
        docs = []
    
        for listing in data["listings"]:
            full_text = self.make_full_text(listing)
            listing_id = compute_listing_id(full_text)
    
            # Skip if already exists
            if listing_id in existing_ids:
                continue
    
            emb = embed_fn(full_text)
    
            item = {
                "id": listing_id,
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
    
        if docs:
            table.add(docs)
            print(f"Added {len(docs)} new listings")
        else:
            print("No new listings to add.")


    def create_vector_index(self, index_type: str = "hnsw", metric: str = "cosine", **kwargs):
        """
        Optional: Create a vector index
        """
        try:
            self.table.create_index("embedding", index_type=index_type, metric=metric, **kwargs)
            print("Created vector index on embedding")
        except Exception as e:
            print("Could not create index (maybe your LanceDB version lacks this API).", e)

    def search_with_lancedb(self, query_vec, limit=5):
        """
        Use native LanceDB search
        Return list of dict records with scrore
        """
        try:
            res = self.table.search(query.vec).limit(limit).to_pandas()
            return res # pandas dataframe with columns including embedding
        except Exception as e:
            print("LanceDB.search() not vailable of failed:", e)
            return None

    def fetch_all_embeddings(self):
        """
        Fallback: return list of embeddings and ids for in-memory similarity
        """
        df = self.table.to_pandas()
        embeddings = df["embedding"].tolist()
        ids = df["id"].tolist()

        return ids, embeddings, df