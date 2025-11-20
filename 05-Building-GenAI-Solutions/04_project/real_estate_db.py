"""
RealEstateDBManager - LanceDB-backed ingestion and search utilities.

Features:
- create/open LanceDB table with a stable schema
- deduplicated ingestion using content MD5 hashes
- optional index creation (guarded for LanceDB versions)
- native LanceDB search + in-memory cosine fallback
- JSON-safe results returned to callers
"""

import hashlib
import json
from typing import List, Tuple, Optional, Any, Dict

import numpy as np
import pandas as pd
import lancedb
from lancedb.pydantic import LanceModel

DEFAULT_EMBED_DIM = int(__import__("os").getenv("EMBED_DIM", "1536"))


def compute_listing_id(full_text: str) -> str:
    """Deterministic id for a listing based on its full_text."""
    return hashlib.md5(full_text.encode("utf-8")).hexdigest()


class RealEstateListing(LanceModel):
    id: str
    neighborhood: str
    price: str
    bedrooms: float
    bathrooms: float
    house_size: str
    description: str
    neighborhood_description: str
    full_text: str
    embedding: List[float]  # vector


class RealEstateDBManager:
    def __init__(self, db_path: str, table_name: str = "real_estate_listing"):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.table = None

    def create_table(self, force: bool = False) -> None:
        """
        Create or load the table. If force=True, drop then recreate.
        """
        if force:
            self.db.drop_table(self.table_name, ignore_missing=True)
            print(f"Dropped old table: {self.table_name}")

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=RealEstateListing)
            print(f"Created new LanceDB table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing LanceDB table: {self.table_name}")

    def get_or_create_table(self):
        if self.table is not None:
            return self.table
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            print(f"Loaded existing table: {self.table_name}")
        else:
            self.table = self.db.create_table(self.table_name, schema=RealEstateListing)
            print(f"Created new table: {self.table_name}")
        return self.table

    def make_full_text(self, raw: dict) -> str:
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
        Ingest listings from JSON while avoiding duplicates.

        json_path: path to a JSON file having structure { "listings": [ ... ] }
        embed_fn: callable(text: str) -> List[float]
        """
        table = self.get_or_create_table()

        # Load existing IDs (fast path)
        try:
            existing_df = table.to_pandas()
            existing_ids = set(existing_df["id"].tolist()) if not existing_df.empty else set()
        except Exception:
            existing_ids = set()

        with open(json_path, "r") as f:
            data = json.load(f)

        docs = []
        for raw in data.get("listings", []):
            full_text = self.make_full_text(raw)
            listing_id = compute_listing_id(full_text)

            if listing_id in existing_ids:
                continue

            emb = embed_fn(full_text)
            if not isinstance(emb, (list, tuple)):
                raise ValueError("embed_fn must return a list of floats")

            docs.append({
                "id": listing_id,
                "neighborhood": raw.get("neighborhood", ""),
                "price": raw.get("price", ""),
                "bedrooms": raw.get("bedrooms", 0),
                "bathrooms": raw.get("bathrooms", 0),
                "house_size": raw.get("house_size", ""),
                "description": raw.get("description", ""),
                "neighborhood_description": raw.get("neighborhood_description", ""),
                "full_text": full_text,
                "embedding": emb,
            })

        if docs:
            table.add(docs)
            print(f"Added {len(docs)} new listings")
        else:
            print("No new listings to add.")

    def create_vector_index(self, index_type: str = "hnsw", metric: str = "cosine", **kwargs):
        """
        Optional: create an index on the embedding column.
        Not all LanceDB versions expose the same API; this is defensive.
        """
        try:
            # Some LanceDB versions accept different signatures; call guarded.
            self.table.create_index("embedding", index_type=index_type, metric=metric, **kwargs)
            print("Created vector index on embedding")
        except TypeError as e:
            # fallback: some versions expect different argnames
            try:
                self.table.create_index("embedding", index_type=index_type, **kwargs)
                print("Created vector index (fallback signature)")
            except Exception as e2:
                print("Could not create index:", e2)
        except Exception as e:
            print("Could not create index (maybe your LanceDB version lacks this API).", e)

    def search_with_lancedb(self, query_vec: List[float], limit: int = 5) -> Optional[pd.DataFrame]:
        """
        Try to use native LanceDB search. Returns a pandas.DataFrame or None.
        """
        try:
            # ensure query_vec is plain python list or numpy array
            res = self.table.search(query_vec).limit(limit).to_pandas()
            return res
        except Exception as e:
            print("LanceDB.native search not available or failed:", e)
            return None

    def fetch_all_embeddings(self) -> Tuple[List[str], List[List[float]], pd.DataFrame]:
        """
        Fallback: return list of ids, embeddings and full DataFrame for in-memory similarity.
        """
        df = self.table.to_pandas()
        ids = df["id"].tolist()
        embeddings = df["embedding"].tolist()
        return ids, embeddings, df

    # Utility to produce JSON-safe dict rows
    @staticmethod
    def sanitize_row_for_json(row: dict) -> dict:
        out = {}
        for k, v in row.items():
            if isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            else:
                out[k] = v
        return out