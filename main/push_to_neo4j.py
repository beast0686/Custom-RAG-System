import os
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm

# --- Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()

# --- MongoDB Connection ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION")

# --- Neo4j Connection ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# --- Main Graph Builder Class ---

class MongoToNeo4jGraphBuilder:
    """
    A class to build a Neo4j Graph from structured documents in MongoDB.
    It maps specific fields to nodes and creates relationships, and also
    calculates semantic similarity between documents using embeddings.
    """

    def __init__(self, mongo_client, neo4j_driver):
        self.mongo_client = mongo_client
        self.db = self.mongo_client[MONGO_DB_NAME]
        self.collection = self.db[MONGO_COLLECTION_NAME]
        self.neo4j_driver = neo4j_driver

    def _execute_cypher_query(self, query, parameters=None):
        """Helper function to execute a Cypher query."""
        # A session should be used for a logical unit of work.
        with self.neo4j_driver.session() as session:
            result = session.run(query, parameters)
            return result.single()

    def _create_constraints(self):
        """Create unique constraints in Neo4j to prevent duplicate nodes."""
        print("Creating Neo4j constraints...")
        self._execute_cypher_query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.mongo_id IS UNIQUE")
        self._execute_cypher_query("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        self._execute_cypher_query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE")
        self._execute_cypher_query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
        self._execute_cypher_query("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")
        print("Constraints created successfully.")

    def process_all_documents(self):
        """
        Fetches all documents from MongoDB and processes them, showing a progress bar.
        """
        print(f"Fetching all documents from '{MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}'...")

        # Get total document count for tqdm progress bar
        total_docs = self.collection.count_documents({})
        documents = self.collection.find()

        doc_embeddings = {}

        # Use tqdm to create a progress bar
        for doc in tqdm(documents, total = total_docs, desc = "Processing Documents"):
            mongo_id = str(doc.get('_id'))
            embedding = doc.get('embedding')  # Note: 'embedding' is the column name in your CSV

            if not embedding:
                # You can log this to a file if needed, but for now we'll just skip
                continue

            # Create nodes and relationships for the current document
            self.create_nodes_from_document(doc)

            # Store embedding for similarity calculation
            doc_embeddings[mongo_id] = embedding

        # Create relationships based on vector similarity
        self.create_similarity_relationships(doc_embeddings)

    def create_nodes_from_document(self, doc):
        """
        Creates a Document node and related entity nodes (Author, Category, etc.)
        from a single MongoDB document.
        """
        # Extract data from the document, providing defaults for missing fields
        mongo_id = str(doc.get('_id'))
        params = {
            "mongo_id": mongo_id,
            "title": doc.get('title', ''),
            "summary": doc.get('summary', ''),
            "url": doc.get('url', ''),
            "date": doc.get('date', ''),
            "embedding": doc.get('embedding', []),
            "author": doc.get('author'),
            "domain": doc.get('domain'),
            "category": doc.get('categories'),  # Note: 'categories' is the column name in your CSV
            "keywords": []
        }

        # Clean up keywords string: split by comma, strip whitespace, remove trailing periods
        keywords_str = doc.get('keywords', '')
        if keywords_str and isinstance(keywords_str, str):
            # Remove trailing period and then split
            cleaned_str = keywords_str.rstrip('. ')
            params["keywords"] = [k.strip() for k in cleaned_str.split(',')]

        # Build a single, robust Cypher query using FOREACH for conditional creation.
        # This fixes the syntax error by ensuring the query is a single, valid statement.
        query = """
        // Create the main Document node with its properties
        MERGE (doc:Document {mongo_id: $mongo_id})
        SET doc.title = $title, 
            doc.summary = $summary, 
            doc.url = $url, 
            doc.date = $date, 
            doc.embedding = $embedding

        WITH doc

        // Conditionally create Author and relationship if author is not null or empty
        FOREACH (_ IN CASE WHEN $author IS NOT NULL AND $author <> '' THEN [1] ELSE [] END |
            MERGE (a:Author {name: $author})
            MERGE (doc)-[:AUTHORED_BY]->(a)
        )

        // Conditionally create Domain and relationship if domain is not null or empty
        FOREACH (_ IN CASE WHEN $domain IS NOT NULL AND $domain <> '' THEN [1] ELSE [] END |
            MERGE (d:Domain {name: $domain})
            MERGE (doc)-[:HOSTED_ON]->(d)
        )

        // Conditionally create Category and relationship if category is not null or empty
        FOREACH (_ IN CASE WHEN $category IS NOT NULL AND $category <> '' THEN [1] ELSE [] END |
            MERGE (c:Category {name: $category})
            MERGE (doc)-[:HAS_CATEGORY]->(c)
        )

        // Create Keyword nodes and relationships from the list, filtering empty strings
        FOREACH (keyword_name IN [k IN $keywords WHERE k IS NOT NULL AND k <> ''] |
            MERGE (k:Keyword {name: keyword_name})
            MERGE (doc)-[:HAS_KEYWORD]->(k)
        )
        """
        self._execute_cypher_query(query, params)

    def create_similarity_relationships(self, doc_embeddings, similarity_threshold=0.9):
        """
        Calculates cosine similarity between all processed documents and creates
        [:SIMILAR_TO] relationships for pairs above a certain threshold.
        """
        print("\n--- Creating Similarity Relationships ---")
        if len(doc_embeddings) < 2:
            print("Not enough document embeddings to compare.")
            return

        doc_ids = list(doc_embeddings.keys())
        # Ensure embeddings are in a consistent format (list of floats)
        embeddings_matrix = np.array([list(map(float, doc_embeddings[doc_id])) for doc_id in doc_ids])

        print(f"Calculating cosine similarity for {len(doc_ids)} documents...")
        sim_matrix = cosine_similarity(embeddings_matrix)

        # Use tqdm for the similarity calculation loop as well
        # Iterate through the upper triangle of the similarity matrix
        for i in tqdm(range(len(doc_ids)), desc = "Calculating Similarities"):
            for j in range(i + 1, len(doc_ids)):
                score = sim_matrix[i, j]
                if score >= similarity_threshold:
                    doc_id1, doc_id2 = doc_ids[i], doc_ids[j]

                    query = """
                    MATCH (d1:Document {mongo_id: $id1})
                    MATCH (d2:Document {mongo_id: $id2})
                    MERGE (d1)-[r:SIMILAR_TO]-(d2)
                    SET r.score = $score
                    """
                    self._execute_cypher_query(query, {"id1": doc_id1, "id2": doc_id2, "score": float(score)})
        print("Similarity analysis complete.")

    def build(self):
        """
        Main method to run the entire graph building process.
        """
        print("Starting graph build process...")
        self._create_constraints()
        self.process_all_documents()
        print("\nGraph build process finished successfully!")


# --- Main Execution Block ---

if __name__ == "__main__":
    mongo_client = None
    neo4j_driver = None
    try:
        print("Connecting to MongoDB...")
        mongo_client = MongoClient(MONGO_URI)
        mongo_client.admin.command('ping')
        print("MongoDB connection successful.")

        print("Connecting to Neo4j Aura...")
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth = (NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        print("Neo4j connection successful.")

        # Instantiate and run the builder
        builder = MongoToNeo4jGraphBuilder(mongo_client, neo4j_driver)
        builder.build()

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up connections
        if mongo_client:
            mongo_client.close()
            print("\nMongoDB connection closed.")
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j connection closed.")
