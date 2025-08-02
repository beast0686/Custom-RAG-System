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


# Load environment variables from .env file
os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()
from together import Together
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
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# --- Local Model Clients ---
print("\n[INFO] Loading local sentence transformer model...")
LOCAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("[INFO] Embedding model loaded successfully.")

# --- In-memory Cache for Comparison ---
comparison_cache = {}


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


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")
    k = int(data.get("k", 10))
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # 1. Fetch documents from MongoDB
    try:
        query_vector = LOCAL_EMBEDDING_MODEL.encode(user_query, normalize_embeddings=True).tolist()
    except Exception as e:
        return jsonify({"error": f"Failed to generate query embedding: {e}"}), 500

    pipeline = [
        {"$vectorSearch": {"index": "embeddings", "path": "embedding", "queryVector": query_vector,
                           "numCandidates": 100, "limit": k}},
        {"$project": {"_id": 1, "content": 1, "title": 1, "summary": 1, "keywords": 1, "url": 1,
                      "score": {"$meta": "vectorSearchScore"}}}
    ]
    try:
        docs = list(mongo_collection.aggregate(pipeline))
    except Exception as e:
        return jsonify({"error": f"Database vector search failed: {e}."}), 500

    retrieved_docs_for_frontend = [{
        "id": str(doc.get("_id")),
        "score": f"{doc.get('score', 0):.4f}",
        "title": doc.get("title", "[No title]"),
        "summary": doc.get("summary", "[No summary]"),
        "keywords": ", ".join(doc.get("keywords", [])) if isinstance(doc.get("keywords"), list) else doc.get("keywords",
                                                                                                             ""),
        "url": doc.get("url", ""),
    } for doc in docs]

    # 2. Prepare a SINGLE batch of text for the LLM
    docs_text_parts = []
    for doc in docs:
        doc_id = str(doc["_id"])
        docs_text_parts.append(
            f"--- Document ID: {doc_id} ---\nTitle: {doc.get('title', '')}\nSummary: {doc.get('summary', '')}")

    docs_text = "\n\n".join(docs_text_parts)

    # 3. LLM Prompting
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    prompt = f"""
    You are a powerful system that extracts a structured knowledge graph from a collection of documents.
    Each document is tagged with a 'Document ID'.
    
    **CRITICAL RULE: The 'source' entity MUST be the entity that PERFORMS the action ('relation') on the 'target' entity. Do not invert the relationship.**
    -   **Correct Example**: If the text is "IBM announced the development of Infoscope", the output MUST be:
        `{{'source': 'IBM', 'relation': 'DEVELOPED', 'target': 'Infoscope'}}`
    -   **Incorrect Example**: Do NOT output `{{'source': 'Infoscope', 'relation': 'DEVELOPED_BY', 'target': 'IBM'}}`. Always make the actor the source.
    
    Your task is to:
    1. Extract a list of named entities. For EACH entity, you MUST specify the ID of the document it came from.
    2. Extract a list of relationships between those entities.
    3. Return a maximum of 15 entities and 20 relationships in total. Focus on the most relevant and interconnected entities across the documents.
    4. Crucially, for every object in the "relationships" list, you MUST ensure that both the 'source' and 'target' entities are also defined in the "entities" list. Do not create relationships that refer to undefined entities.
    Output ONLY a single, valid JSON object with two keys: "entities" and "relationships".

    The "entities" list must contain objects with THREE keys:
    - "name": The name of the entity.
    - "type": The entity type (e.g., "Person", "Organization", "Technology").
    - "source_document_id": The ID of the document where this entity was found.

    Example output format:
    {{
      "entities": [
        {{ "name": "Alan Turing", "type": "Person", "source_document_id": "668808b86e..." }},
        {{ "name": "Enigma", "type": "Technology", "source_document_id": "668808b86e..." }},
        {{ "name": "IBM", "type": "Organization", "source_document_id": "668808b73c..." }}
      ],
      "relationships": [
        {{ "source": "Alan Turing", "relation": "cracked", "target": "Enigma" }}
      ]
    }}
    Relationships part of the JSON must contain entities from what you have defined in the "entities" list.
    Absolutely do not create relationships that refer to entities not defined in the "entities" list.
    It is fine if you do not find the required number of relationships, but if you do, ensure they are valid.
    Text to extract from:
    \"\"\"
    {docs_text}
    \"\"\"
    """
    try:
        print("[INFO] Sending single batch prompt to Together AI...")
        response = client.chat.completions.create(
            model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}]
        )
        parsed_output = extract_json_from_string(response.choices[0].message.content)
        print(parsed_output)
        if not parsed_output or "entities" not in parsed_output:
            raise ValueError("LLM response was empty or malformed.")
        print("[INFO] Successfully parsed LLM response.")
    except Exception as e:
        return jsonify({"error": f"Knowledge extraction failed: {e}"}), 500

    # 3.5 Generate paragraph answer
    try:
        context_text = "\n\n".join([
            f"Title: {doc.get('title', '[No title]')}\nSummary: {doc.get('summary', '[No summary]')}\nKeywords: {', '.join(doc.get('keywords', []))}"
            for doc in docs
        ])
        answer_prompt = f"""
        You are a helpful assistant that answers user queries using the provided documents.
        Be concise and accurate. If the documents do not provide enough information to fully answer the query,
        you should clearly state what is known and mention that the current RAG system only contains 30,000 documents and cannot fully support your query.
        Query: {user_query}
        Documents:
        {context_text}
        Answer the query using the above documents. Your first 3-5 sentences should directly answer the query.
        Then, provide a paragraph long summary cum explanation of the most relevant documents used to answer the query.
        Do not exceed 150 words.
        Refer to the number and ID's of documents used in your answer. Be clear about this and show it explicitly at the end of your answer as references.
        Do not refer to the documents while providing the direct answer.
        """
        client2 = OpenAI(api_key=baseten_api_key, base_url="https://inference.baseten.co/v1")
        response = client2.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[{"role": "user", "content": answer_prompt}],
            max_tokens=1000,
        )
        paragraph_answer = response.choices[0].message.content
    except Exception as e:
        paragraph_answer = f"[LLM answer generation failed: {e}]"

    # 4. Build precise graph structure
    session_id = str(uuid.uuid4())
    nodes, edges = [], []
    unique_nodes = {}


    referenced_doc_ids = {ent.get("source_document_id") for ent in parsed_output.get("entities", [])}
    for doc_data in retrieved_docs_for_frontend:
        if doc_data['id'] in referenced_doc_ids:
            doc_node_id = f"doc_{doc_data['id']}"
            unique_nodes[doc_node_id] = {"id": doc_node_id, "label": f"Doc: {doc_data['id'][:8]}...",
                                         "group": "Document", "score": float(doc_data['score'])}

    all_entities_from_llm = parsed_output.get("entities", [])
    for ent in all_entities_from_llm:
        if not all(k in ent for k in ["name", "type", "source_document_id"]): continue
        ent_id = f"{ent['type']}_{ent['name']}".replace(" ", "_").lower()
        doc_id = ent["source_document_id"]
        doc_node_id = f"doc_{doc_id}"
        if ent_id not in unique_nodes:
            unique_nodes[ent_id] = {"id": ent_id, "label": ent["name"], "group": ent["type"]}
        if doc_node_id in unique_nodes:
            edges.append({"from": doc_node_id, "to": ent_id})

    # MODIFIED SECTION: Dynamically create nodes for hallucinated entities
    entity_map = {e['name']: f"{e['type']}_{e['name']}".replace(" ", "_").lower() for e in all_entities_from_llm}
    for rel in parsed_output.get("relationships", []):
        src_name = rel.get("source")
        tgt_name = rel.get("target")

        if not src_name or not tgt_name:
            continue

        for entity_name in [src_name, tgt_name]:
            if entity_name not in entity_map:
                print(f"[INFO] Hallucinated entity found: '{entity_name}'. Creating node.")
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', entity_name)
                new_id = f"inferred_{safe_name}".lower()
                if new_id not in unique_nodes:
                    unique_nodes[new_id] = {"id": new_id, "label": entity_name, "group": "Inferred"}
                entity_map[entity_name] = new_id

        src_id = entity_map[src_name]
        tgt_id = entity_map[tgt_name]
        edges.append({"from": src_id, "to": tgt_id, "relation": rel.get("relation", "RELATED_TO")})
    # END OF MODIFIED SECTION

    nodes = list(unique_nodes.values())

    # 5. Push graph to Neo4j using MERGE and ON CREATE SET
    with neo4j_driver.session() as session:
        for node_data in nodes:
            if node_data['group'] == 'Document':
                session.run("""
                    MERGE (d:Document {id: $id})
                    ON CREATE SET d.title = $label, d.session = $sid
                """, id=node_data['id'], label=node_data['label'], sid=session_id)
            else:
                safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', node_data['group'])
                session.run(f"""
                    MERGE (e:{safe_label} {{id: $id}})
                    ON CREATE SET e.name = $label, e.session = $sid
                """, id=node_data['id'], label=node_data['label'], sid=session_id)

        session.run("""
            MERGE (c:Center {id: "db"})
            ON CREATE SET c.label = "DB"
        """)
        for node_data in nodes:
            if node_data["group"] == "Document":
                session.run("""
                    MATCH (c:Center {id: "db"}), (d:Document {id: $doc_id})
                    MERGE (c)-[:CONTAINS]->(d)
                """, doc_id=node_data["id"])

        for edge_data in edges:
            if edge_data.get('relation'):
                rel_type = re.sub(r'[^a-zA-Z0-9_]', '', edge_data['relation'].replace(" ", "_").upper())
                if rel_type:
                    session.run(f"""
                        MATCH (a {{id: $src}}), (b {{id: $tgt}})
                        MERGE (a)-[r:{rel_type}]->(b)
                    """, src=edge_data['from'], tgt=edge_data['to'])
            else:
                session.run("""
                    MATCH (d:Document {id: $src}), (e {id: $tgt})
                    MERGE (d)-[:MENTIONS]->(e)
                """, src=edge_data['from'], tgt=edge_data['to'])
    all_entity_names = list(entity_map.keys())
    comparison_cache[session_id] = {
        "query": user_query,
        "docs": docs,
        "mongodb_rag_answer": paragraph_answer,
        "extracted_entities": all_entity_names,
        "document_info": retrieved_docs_for_frontend
    }
    return jsonify({
        "retrieved_docs": retrieved_docs_for_frontend,
        "nodes": nodes,
        "edges": edges,
        "session_id": session_id,
        "answer": paragraph_answer or "No answer generated."
    })

@app.route("/generate_comparison", methods=["POST"])
def generate_comparison():
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id or session_id not in comparison_cache:
        return jsonify({"error": "Invalid or expired session ID."}), 404

    cached_data = comparison_cache[session_id]
    user_query = cached_data["query"]
    docs = cached_data["docs"]

    # --- Initialize Baseten Client ---
    baseten_client = OpenAI(api_key=baseten_api_key, base_url="https://inference.baseten.co/v1")
    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    # --- 1. Plain LLM Answer ---
    try:
        plain_prompt = f"""
        You are a helpful assistant. Answer the following user query directly based on your general knowledge.
        Be concise and do not exceed 150 words. Do not exceed a paragraph.
        Do not use bold or italic formatting. Keep the answer plain-text and simple.
        Query: {user_query}
        """
        response = baseten_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": plain_prompt}],
            max_tokens=1000,
        )
        plain_llm_answer = response.choices[0].message.content
    except Exception as e:
        plain_llm_answer = f"[Plain LLM answer generation failed: {e}]"

    # --- 2. MongoDB Vector Search Based LLM Answer (from cache) ---
    mongodb_rag_answer = cached_data.get("mongodb_rag_answer", "[Answer not found in cache]")

    # --- 3. Neo4j + LLM Answer---
    try:
        # Step 3.1: Get the pre-extracted entity list from the cache
        entities = cached_data.get("extracted_entities")
        print(f"[INFO] Using pre-extracted entities for KG search: {entities}")

        if not entities:
            raise ValueError("No entities were pre-extracted from the documents.")
        info = cached_data.get("document_info")
        # Step 3.2: Query Neo4j to get the neighborhood of these entities
        kg_context_triples = []
        with neo4j_driver.session() as session:
            cypher_query = """
                UNWIND $entities AS entityName
                MATCH (n) WHERE n.name CONTAINS entityName
                MATCH (n)-[r]->(m)
                WHERE m.name IS NOT NULL
                RETURN n.name AS source, type(r) AS relation, m.name AS target
                LIMIT 25
            """
            results = session.run(cypher_query, entities = entities)
            for record in results:
                kg_context_triples.append(f"({record['source']})-[:{record['relation']}]->({record['target']})")

        kg_context = "\n".join(kg_context_triples)
        print(f"[INFO] Retrieved KG context:\n{kg_context}")

        if not kg_context:
            neo4j_kg_rag_answer = "The entities extracted from the documents were not found in the knowledge graph."
        else:
            # Step 3.3: Pass the KG context to the LLM to generate an answer
            kg_rag_prompt = f"""
                You are an assistant that answers queries using the provided Knowledge Graph context.
                Answer the user's query using ONLY the facts from the context below. Do not use any prior knowledge.
                If the context does not contain the answer, say so.
                Keep the answer concise and do not exceed 150 words and one paragraph only. Do not use bold or italic formatting.
                Take the kg context into account and generate an elaborate answer.
                DO not mention about what you know and do not - use abstraction and professionalism.
                
                Do not explicitly mention "context" or "knowledge graph" or "mentioned entities" in your answer.
                Use the information provided to refer to the entities and relationships without explicitly stating them.
                Query: {user_query}

                Knowledge Graph Context:
                ---
                {kg_context}
                ---
                
                Document Information:
                ---
                {info}
                ---

                Answer:
                """
            response = baseten_client.chat.completions.create(
                model = model_name,
                messages = [{"role": "user", "content": kg_rag_prompt}],
                max_tokens = 1000,
            )
            neo4j_kg_rag_answer = response.choices[0].message.content

    except Exception as e:
        neo4j_kg_rag_answer = f"[KG RAG answer generation failed: {e}]"
        print(f"[ERROR] KG RAG answer generation failed: {e}")

    return jsonify({
        "plain_llm_answer": plain_llm_answer,
        "mongodb_rag_answer": mongodb_rag_answer,
        "neo4j_kg_rag_answer": neo4j_kg_rag_answer
    })


@app.route("/cleanup/<session_id>", methods=['POST'])
def cleanup(session_id):
    with neo4j_driver.session() as session:
        session.run("MATCH (n {session: $sid}) DETACH DELETE n", sid=session_id)
    # Also remove from cache
    if session_id in comparison_cache:
        del comparison_cache[session_id]
    return f"Cleanup complete for session {session_id}"


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)