import json
import lancedb
from lancedb.pydantic import LanceModel
from typing import List


class RealEstateListing(LanceModel):
    neighborhood: str
    price: str
    bedrooms: float
    bathrooms: float
    house_size: str
    description: str


class RealEstateDBManager:
    def __init__(self, db_path: str, table_name: str = "real_estate_listing"):
        """Initialize LanceDB connection and table name."""
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(db_path)
        self.table = None

    def create_table(self):
        """Create or recreate the LanceDB table."""
        self.db.drop_table(self.table_name, ignore_missing=True)
        self.table = self.db.create_table(self.table_name, schema=RealEstateListing)
        print(f"✅ Table '{self.table_name}' created successfully.")

    def load_listings_from_json(self, json_path: str) -> List[RealEstateListing]:
        """Load listings from a JSON file."""
        with open(json_path, 'r') as file:
            data = json.load(file)

        listings = [
            RealEstateListing(
                neighborhood=listing['neighborhood'],
                price=listing['price'],
                bedrooms=listing['bedrooms'],
                bathrooms=listing['bathrooms'],
                house_size=listing['house_size'],
                description=listing['description']
            )
            for listing in data['listings']
        ]
        print(f"📦 Loaded {len(listings)} listings from {json_path}.")
        return listings

    def add_listings(self, listings: List[RealEstateListing]):
        """Add listings to the LanceDB table, creating it if missing."""
        if not self.table:
            print("⚠️ Table not found. Creating table automatically...")
            self.create_table()
        self.table.add([dict(l) for l in listings])
        print(f"✅ Added {len(listings)} listings to the '{self.table_name}' table.")


    def setup_from_json(self, json_path: str):
        """Convenience method to create the table and load data in one go."""
        self.create_table()
        listings = self.load_listings_from_json(json_path)
        self.add_listings(listings)
   
    def print_head(self):
        """Prints the head"""
        print(self.table.head().to_pandas())
        
    def drop_head(self):
        """Drop table"""
        self.db.drop_table(self.table_name)
        print("Table dropped", self.table.name in self.db)


if __name__ == "__main__":
    db_path = "../../../../data/GenAI/05_project/"
    json_path = "real_estate_listing.json"

    manager = RealEstateDBManager(db_path)
    manager.setup_from_json(json_path)
