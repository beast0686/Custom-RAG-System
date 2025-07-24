import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import requests

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
collection = client[MONGO_DB][MONGO_COLLECTION]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# User input
sentence = input("🔎 Enter your query sentence: ").strip()
embedding_vector = model.encode(sentence, normalize_embeddings=True).tolist()

# Vector search in MongoDB
pipeline = [
    {
        "$vectorSearch": {
            "index": "embeddings",
            "path": "embedding",
            "queryVector": embedding_vector,
            "numCandidates": 100,
            "limit": 5
        }
    }
]
results = list(collection.aggregate(pipeline))
print(f"\n📄 Retrieved {len(results)} documents for: \"{sentence}\"\n")

# Build context from results
context_snippets = []
for doc in results:
    title = doc.get("title", "Untitled")
    date = doc.get("date", "Unknown")
    content = doc.get("text") or doc.get("full_text") or doc.get("abstract") or ""
    snippet = f"Title: {title}\nDate: {date}\nContent: {content.strip()}"
    context_snippets.append(snippet)

combined_context = "\n\n---\n\n".join(context_snippets)

# Prompt with documents
prompt_with_docs = f"""
You are a senior research analyst working on synthesizing findings across multiple knowledge sources. You are given 5 documents that were retrieved from a vector similarity search engine for the query:

"{sentence}"

These documents may include reports, academic abstracts, internal notes, or market observations. Your goal is to synthesize the *key ideas* expressed in these documents and produce a unified, coherent, and sophisticated paragraph.

### Constraints:

- Only use information explicitly stated in the documents below.
- Do NOT add any external knowledge, assumptions, or hallucinated facts.
- Avoid bullet points or listing each document separately.
- Do NOT refer to document titles or metadata (e.g., “Document 1 says…”).
- Maintain a neutral, analytical tone.
- Prioritize conceptual coherence and interpretive synthesis over summarization.

### Output:

Write a single, well-structured paragraph that:
- Combines themes or patterns across documents.
- Identifies nuanced insights, emerging trends, or common conclusions.
- Reflects critical analysis as if written by a domain expert.

### Documents:
{combined_context}

### Response:
"""

# Prompt without documents
prompt_without_docs = f"""
You are a synthesis expert. Write a thoughtful, insightful paragraph about the topic: "{sentence}".
Do not use the documents and just treat it as an independant prompt and generate it seperately. It is merely used for comparasion purposes.
Note: It is very important you treat this as a seperate query and do not include the documents mentioned in other prompts since it is used for comparision purposes.
"""

# Helper function to call Perplexity
def call_perplexity(prompt):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"❌ API Error ({response.status_code}): {response.text}")
        return None

# Get responses
print("\n🧠 Generating LLM synthesis WITH document context...")
with_docs_output = call_perplexity(prompt_with_docs)

print("\n🧠 Generating LLM synthesis WITHOUT document context...")
without_docs_output = call_perplexity(prompt_without_docs)

# Display results
if with_docs_output:
    print("\n🔍 LLM-Synthesized Insight (WITH Documents):\n")
    print(with_docs_output)

if without_docs_output:
    print("\n🧠 LLM-Synthesized Insight (WITHOUT Documents):\n")
    print(without_docs_output)
