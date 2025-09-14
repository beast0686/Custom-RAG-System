import os
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase
from tqdm import tqdm
import time

# --- Configuration and Initialization ---

# Load environment variables from .env file
# Make sure your .env file is configured for your LOCAL Neo4j instance.
# Example .env content:
# MONGO_URI="mongodb://localhost:27017/"
# MONGO_DB="your_mongo_db"
# MONGO_COLLECTION="your_mongo_collection"
#
# # --- Local Neo4j Connection Settings ---
# NEO4J_URI="bolt://localhost:7687"
# NEO4J_USERNAME="neo4j"
# NEO4J_PASSWORD="your_local_db_password"

load_dotenv()

# --- MongoDB Connection ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION")

# --- Neo4j Connection ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class MongoToNeo4jIngestor:
    """
    A class focused solely on ingesting documents from MongoDB into a local Neo4j
    database as individual :Document nodes. It includes a function to clear the database.
    """

    def __init__(self, mongo_client, neo4j_driver):
        self.mongo_client = mongo_client
        self.db = self.mongo_client[MONGO_DB_NAME]
        self.collection = self.db[MONGO_COLLECTION_NAME]
        self.neo4j_driver = neo4j_driver

    def clear_database(self):
        """
        Deletes all nodes and relationships from the Neo4j database in batches
        to prevent timeouts on large databases.
        """
        print("\n" + "=" * 50)
        print("WARNING: Preparing to delete ALL data from the Neo4j database.")
        print("This action cannot be undone.")
        print("=" * 50)

        # Countdown to give the user a chance to cancel
        for i in range(5, 0, -1):
            print(f"  Starting in {i} seconds...", end='\r')
            time.sleep(1)

        print("\nClearing database in batches...")

        # Loop to delete nodes in batches until none are left
        while True:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WITH n LIMIT 10000
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                """)
                deleted_count = result.single()["deleted_count"]
                if deleted_count > 0:
                    print(f"  Deleted a batch of {deleted_count} nodes...")
                else:
                    print("  No more nodes to delete.")
                    break

        print("Neo4j database has been cleared.")

    def create_constraints(self):
        """Create a unique constraint on the Document node's mongo_id."""
        print("Creating Neo4j constraint for :Document(mongo_id)...")
        with self.neo4j_driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.mongo_id IS UNIQUE")
        print("Constraint created successfully.")

    def ingest_all_documents(self):
        """
        Fetches all documents from MongoDB and creates a corresponding :Document node
        in Neo4j for each one.
        """
        print(f"Starting ingestion from '{MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}'...")

        total_docs = self.collection.count_documents({})
        if total_docs == 0:
            print("No documents found in the collection. Exiting.")
            return

        documents_cursor = self.collection.find()

        ingest_query = """
        MERGE (d:Document {mongo_id: $mongo_id})
        SET d += $props
        """

        with self.neo4j_driver.session() as session:
            for doc in tqdm(documents_cursor, total=total_docs, desc="Ingesting Documents"):
                props_to_set = {
                    "title": doc.get('title'),
                    "content": doc.get('content'),
                    "url": doc.get('url'),
                    "domain": doc.get('domain'),
                    "author": doc.get('author'),
                    "categories": doc.get('categories'),
                    "summary": doc.get('summary'),
                    "keywords": doc.get('keywords'),
                    "embedding": doc.get('embedding')
                }

                # Remove keys with None values to avoid storing them in Neo4j
                props_to_set = {k: v for k, v in props_to_set.items() if v is not None}

                params = {
                    "mongo_id": str(doc['_id']),
                    "props": props_to_set
                }
                session.run(ingest_query, params)

        print("\nDocument ingestion completed successfully!")

    def run_ingestion(self):
        """Main method to run the entire ingestion process."""
        self.clear_database()
        self.create_constraints()
        self.ingest_all_documents()


# --- Main Execution Block ---

if __name__ == "__main__":
    mongo_client = None
    neo4j_driver = None
    try:
        print("Connecting to MongoDB...")
        mongo_client = MongoClient(MONGO_URI)
        # The ismaster command is cheap and does not require auth.
        mongo_client.admin.command('ping')
        print("MongoDB connection successful.")

        print("Connecting to local Neo4j database...")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print("Neo4j connection successful.")

        ingestor = MongoToNeo4jIngestor(mongo_client, neo4j_driver)
        ingestor.run_ingestion()

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
            print("\nMongoDB connection closed.")
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j connection closed.")
