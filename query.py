import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# Sample Queries

print("\n🔍 1. Find one document:")
print(collection.find_one())

print("\n🔍 2. Find all documents with a specific title:")
for doc in collection.find({"title": "From takeoff to flight, the wiring of a fly's nervous system is mapped"}):
    print(doc)

print("\n🔍 3. Find documents with date after 2024-01-01:")
for doc in collection.find({"date": {"$gt": datetime(2024, 1, 1)}}).limit(5):
    print(doc)

print("\n🔍 4. Count total documents:")
print(collection.count_documents({}))

print("\n🔍 5. Find document with a specific embedding value:")
sample_query_value = 0.123456  # replace with real value if needed
for doc in collection.find({"embedding": sample_query_value}):
    print(doc)
