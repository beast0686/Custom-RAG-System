import os
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

# --- Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)

# --- MongoDB Connection ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION")
mongo_client = MongoClient(MONGO_URI)
mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]

# --- Neo4j Connection ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth = (NEO4J_USER, NEO4J_PASSWORD))

# --- Sentence Transformer Model ---
# This will download the model on the first run.
print("Loading Sentence Transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")


# --- Backend API Endpoints ---

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/search', methods = ['POST'])
def search():
    """
    1. Takes a user query.
    2. Creates an embedding for the query.
    3. Performs vector search in MongoDB.
    4. Fetches the corresponding subgraph from Neo4j.
    5. Returns the graph data as JSON.
    """
    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # 1. Create embedding for the user query
    embedding_vector = model.encode(user_query, normalize_embeddings = True).tolist()

    # 2. Perform vector search in MongoDB
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embeddings",
                "path": "embedding",
                "queryVector": embedding_vector,
                "numCandidates": 100,
                "limit": 5
            }
        },
        {
            "$project": {
                "_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    try:
        mongo_results = list(mongo_collection.aggregate(pipeline))
        if not mongo_results:
            return jsonify({"nodes": [], "links": []})

        # Extract the MongoDB document IDs
        mongo_ids = [str(doc['_id']) for doc in mongo_results]

        # 3. Fetch the subgraph from Neo4j using the retrieved IDs
        with neo4j_driver.session() as session:
            # FIX: Changed read_transaction to the recommended execute_read
            graph_data = session.execute_read(get_subgraph_for_ids, mongo_ids)

        return jsonify(graph_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


def get_subgraph_for_ids(tx, mongo_ids):
    """
    Cypher query to get all nodes and relationships connected to the given Document IDs.
    """
    # FIX: Replaced the complex CALL {} subquery with a simpler OPTIONAL MATCH.
    # This resolves the "Variable `d` already declared" error and is more efficient.
    query = """
    MATCH (d:Document)
    WHERE d.mongo_id IN $mongo_ids
    OPTIONAL MATCH (d)-[r]-(n)
    RETURN d, r, n
    """
    results = tx.run(query, mongo_ids = mongo_ids)

    nodes = {}
    links = []

    for record in results:
        # Process the document node
        d_node = record['d']
        d_id = d_node.id
        if d_id not in nodes:
            nodes[d_id] = {
                "id": d_id,
                "label": list(d_node.labels)[0],
                "name": d_node.get('title', d_node.get('name', 'Document'))
            }

        # Process the related node (if it exists)
        n_node = record['n']
        if n_node:
            n_id = n_node.id
            if n_id not in nodes:
                nodes[n_id] = {
                    "id": n_id,
                    "label": list(n_node.labels)[0],
                    "name": n_node.get('name', 'N/A')
                }

        # Process the relationship (if it exists)
        r_rel = record['r']
        if r_rel:
            links.append({
                "source": r_rel.start_node.id,
                "target": r_rel.end_node.id,
                "type": r_rel.type
            })

    # Remove duplicate links that might arise from the query
    unique_links = [dict(t) for t in {tuple(d.items()) for d in links}]

    return {"nodes": list(nodes.values()), "links": unique_links}


# --- Main Execution Block ---

if __name__ == "__main__":
    # Note: Use 'waitress' for production instead of app.run()
    app.run(debug = True, port = 5001)

