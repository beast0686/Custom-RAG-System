import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

client = MongoClient(MONGO_URI)
collection = client[MONGO_DB][MONGO_COLLECTION]

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "The future of artificial intelligence is promising."
embedding_vector = model.encode(sentence, normalize_embeddings=True).tolist()

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
print(f"\n🔍 Top {len(results)} results for sentence: \"{sentence}\"\n")

for i, doc in enumerate(results, 1):
    print(f"Result #{i}")
    print(f"Title: {doc.get('title', 'N/A')}")
    print(f"Date: {doc.get('date', 'N/A')}")
    print(f"ID: {doc.get('_id')}")
    print(f"Score (if available): {doc.get('score', 'N/A')}")
    print("-" * 60)
