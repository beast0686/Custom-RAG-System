from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from neo4j import GraphDatabase
import requests
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import load_dotenv
import os
import re
import json
from openai import OpenAI

# --- InsightGraph: Knowledge Graph Extraction Service ---

# Load environment variables from .env file
load_dotenv()
from together import Together

client = Together(api_key = os.getenv("TOGETHER_API_KEY"))
# --- Initializations ---
app = Flask(__name__)
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB")
mongo_collection_name = os.getenv("MONGO_COLLECTION")
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client[mongo_db_name]
mongo_collection = mongo_db[mongo_collection_name]
baseten_api_key = os.getenv("BASETEN_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth = (neo4j_user, neo4j_password))

# --- Local Model Clients ---
print("\n[INFO] Loading local sentence transformer model...")
LOCAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("[INFO] Embedding model loaded successfully.")


# --- Utility Functions ---
def extract_json_from_string(s):
    try:
        return json.loads(s.strip())
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        print(f"[DEBUG] Raw content received:\n{s}")
        return None


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods = ["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")
    k = int(data.get("k", 10))  # default to 10 if not provided
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # 1. Generate embedding for the user query with NORMALIZATION
    try:
        query_vector = LOCAL_EMBEDDING_MODEL.encode(user_query, normalize_embeddings = True).tolist()
    except Exception as e:
        return jsonify({"error": f"Failed to generate local query embedding: {e}"}), 500

    # 2. Fetch top K similar documents from MongoDB
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embeddings",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {"$project": {
            "content": 1,
            "title": 1,
            "author": 1,
            "summary": 1,
            "keywords": 1,
            "url": 1,
            "score": {"$meta": "vectorSearchScore"}
        }}
    ]
    try:
        docs = list(mongo_collection.aggregate(pipeline))
    except Exception as e:
        return jsonify({"error": f"Database vector search failed: {e}."}), 500
    print(docs)
    retrieved_docs_for_frontend = []
    for doc in docs:
        retrieved_docs_for_frontend.append({
            "id": str(doc.get("_id")),
            "score": f"{doc.get('score', 0):.4f}",
            "title": doc.get("title", "[No title]"),
            "author": doc.get("author", "[Unknown]"),
            "summary": doc.get("summary", "[No summary]"),
            "keywords": ", ".join(doc.get("keywords", [])) if isinstance(doc.get("keywords"), list) else doc.get(
                "keywords", ""),
            "url": doc.get("url", ""),
            "content_snippet": doc.get("content", "")[:350] + "..."
        })

    # 3. Batch documents and send to Ollama
    session_id = str(uuid.uuid4())
    INCLUDE_FIELDS = {"title", "author", "summary", "keywords", "url", "score"}

    def clean_doc(doc):
        return {
            k: ", ".join(v) if isinstance(v, list) else str(v)
            for k, v in doc.items()
            if k in INCLUDE_FIELDS and v  # skip empty values
        }

    docs_text = "\n\n--- Document ---\n\n".join([
        "\n".join(f"{k}: {v}" for k, v in clean_doc(doc).items())
        for doc in retrieved_docs_for_frontend
    ])

    prompt = f"""
    You are a powerful system that extracts structured knowledge from text.

    Your task is to extract:
    1. A list of named entities, each with:
       - "name": the entity name
       - "type": one of ["Person", "Organization", "Location", "Concept", "Technology", "Event", "Product", "Other"]

    2. A list of relationships between entities, each with:
       - "source": name of the source entity
       - "relation": a concise verb or phrase describing the relationship
       - "target": name of the target entity

    3. You must only return not more than 10 entities and 20 relationships for each in total no matter how many documents. These should be the most relevant ones based on the provided documents.
    These should also make sense and be interlinked amongst documents. 

    Output ONLY a valid JSON object with two top-level keys: "entities" and "relationships".

    - Do not explain anything.
    - Do not include markdown, comments, or code block markers (like ```json).
    - Do not include any text outside the JSON.
    - Make sure the JSON is syntactically correct and can be parsed directly.

    Example output format:

    {{
      "entities": [
        {{ "name": "Alan Turing", "type": "Person" }},
        {{ "name": "Enigma", "type": "Technology" }}
      ],
      "relationships": [
        {{ "source": "Alan Turing", "relation": "developed", "target": "Enigma" }}
      ]
    }}

    Text to extract from:
    \"\"\"
    {docs_text}
    \"\"\"
    """

    try:
        print(f"Prompt sent to Ollama:\n{docs_text}\n")
        response = client.chat.completions.create(
            model = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        print(response)
        parsed_output = extract_json_from_string(response.choices[0].message.content)
        print("\n\n")
        print(parsed_output)
    except Exception as e:
        return jsonify({"error": f"Ollama extraction failed: {e}"}), 500

    if not parsed_output or "entities" not in parsed_output:
        return jsonify({"error": "Entity/relationship extraction failed or returned invalid format"}), 500

    # 3.5 Generate paragraph answer to user query using Baseten LLM
    try:
        context_text = "\n\n".join([
            f"Title: {doc['title']}\nSummary: {doc['summary']}\nKeywords: {doc['keywords']}"
            for doc in retrieved_docs_for_frontend
        ])

        answer_prompt = f"""
        You are a helpful assistant that answers user queries using the provided documents.
        Be concise and accurate. If the documents do not provide enough information to fully answer the query,
        you should clearly state what is known and mention that the current RAG system only contains 30,000 documents and cannot fully support your query.

        Query: {user_query}

        Documents:
        {context_text}

        Answer the query using the above documents.
        Output format and instructions:
        Your first two sentences should directly answer the query.
        Then, provide a paragraph long summary cum explanation of the most relevant documents used to answer the query.
        Use the following format:
        Plain text only - Avoid bold, italics, or any other formatting.
        Do not include any markdown, code blocks, or explanations.
        Keep the answer concise and relevant to the query.
        Do not exceed 100 words.
        Refer to the number and ID's of documents used in your answer. Be clear about this and show it explicitly at the end of your answer as references.
        You do not have to use all documents, only the most relevant ones.
        """
        client2 = OpenAI(
            api_key = baseten_api_key,
            base_url = "https://inference.baseten.co/v1"
        )

        response = client2.chat.completions.create(
            model = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages = [
                {"role": "user", "content": answer_prompt}
            ],
            stream = True,
            stream_options = {
                "include_usage": True,
                "continuous_usage_stats": True
            },
            max_tokens = 500,
            temperature = 0.9,
        )

        paragraph_answer = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                paragraph_answer += chunk.choices[0].delta.content

    except Exception as e:
        paragraph_answer = f"[LLM answer generation failed: {e}]"

    # 4. Push entities, relationships, and document links into Neo4j
    nodes, edges, seen_nodes, seen_nodes_neo4j = [], [], set(), set()

    with neo4j_driver.session() as session:
        # Insert document nodes
        for doc in docs:
            doc_id = str(doc["_id"])
            doc_node_id = f"doc_{doc_id}"

            if doc_node_id not in seen_nodes_neo4j:
                session.run(
                    "MERGE (d:Document {id: $id, session: $sid}) SET d.title = $title",
                    {"id": doc_node_id, "sid": session_id, "title": f"Doc: {doc_id[:8]}..."}
                )
                seen_nodes_neo4j.add(doc_node_id)

            if doc_node_id not in seen_nodes:
                nodes.append({"id": doc_node_id, "label": f"Doc: {doc_id[:8]}...", "group": "document"})
                seen_nodes.add(doc_node_id)

        # Insert entity nodes
        for ent in parsed_output.get("entities", []):
            if 'name' not in ent or 'type' not in ent:
                continue

            ent_id = f"{ent['type']}_{ent['name']}".replace(" ", "_").lower()
            label = ent['type'].capitalize()

            if ent_id not in seen_nodes:
                # First check if node already exists in Neo4j
                result = session.run(
                    "MATCH (e {id: $id}) RETURN e LIMIT 1", {"id": ent_id}
                )
                if not result.peek():  # Only create if not found
                    session.run(
                        f"CREATE (e:{label} {{id: $id, session: $sid, name: $name, type: $type}})",
                        {"id": ent_id, "sid": session_id, "name": ent["name"], "type": ent["type"]}
                    )
                # Add to local node list for visualization
                nodes.append({"id": ent_id, "label": ent["name"], "group": ent["type"]})
                seen_nodes.add(ent_id)

        # Insert document-to-entity edges
        for doc in docs:
            doc_id = str(doc["_id"])
            doc_node_id = f"doc_{doc_id}"
            for ent in parsed_output.get("entities", []):
                ent_id = f"{ent['type']}_{ent['name']}".replace(" ", "_").lower()
                session.run(
                    "MATCH (d:Document {id: $doc_id, session: $sid}), "
                    "(e {id: $ent_id, session: $sid}) "
                    "MERGE (d)-[:MENTIONS]->(e)",
                    {"doc_id": doc_node_id, "ent_id": ent_id, "sid": session_id}
                )
                edges.append({"from": doc_node_id, "to": ent_id})

        # Insert entity-to-entity relationships
        for rel in parsed_output.get("relationships", []):
            src = rel.get("source")
            tgt = rel.get("target")
            rel_type = rel.get("relation", "RELATED_TO").replace(" ", "_").upper()
            if not src or not tgt:
                continue

            src_id = None
            tgt_id = None

            for ent in parsed_output["entities"]:
                if ent["name"] == src:
                    src_id = f"{ent['type']}_{ent['name']}".replace(" ", "_").lower()
                if ent["name"] == tgt:
                    tgt_id = f"{ent['type']}_{ent['name']}".replace(" ", "_").lower()

            if src_id and tgt_id:
                session.run(
                    "MATCH (a {id: $src_id, session: $sid}), (b {id: $tgt_id, session: $sid}) "
                    f"MERGE (a)-[r:{rel_type}]->(b)",
                    {"src_id": src_id, "tgt_id": tgt_id, "sid": session_id}
                )
                edges.append({
                    "from": src_id,
                    "to": tgt_id,
                    "relation": rel.get("relation", "RELATED_TO")  # ✅ Add this
                })

    return jsonify({
        "retrieved_docs": retrieved_docs_for_frontend,
        "nodes": nodes,
        "edges": edges,
        "session_id": session_id,
        "answer": paragraph_answer or "No answer generated."
    })


@app.route("/cleanup/<session_id>", methods = ['POST'])
def cleanup(session_id):
    with neo4j_driver.session() as session:
        session.run("MATCH (n {session: $sid}) DETACH DELETE n", sid = session_id)
    return f"Cleanup complete for session {session_id}"


if __name__ == '__main__':
    app.run(debug = True, port = 5001)
