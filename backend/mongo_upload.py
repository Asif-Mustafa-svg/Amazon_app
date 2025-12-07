import json
from pymongo import MongoClient, InsertOne
from tqdm import tqdm

BATCH_SIZE = 1000

# --------------------------
# 1. MongoDB Atlas Connection
# --------------------------
client = MongoClient(
    "I have removed this."
)

db = client["amazon_reviews"]
col = db["combined_reviews"]

# --------------------------
# 2. Load JSONL
# --------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# --------------------------
# 3. Bulk Insert Function
# --------------------------
def bulk_insert_jsonl(path, collection, batch_size=BATCH_SIZE):
    buffer = []
    count = 0

    print("⬆️ Uploading data into MongoDB Atlas...\n")

    for doc in tqdm(load_jsonl(path)):
        buffer.append(InsertOne(doc))

        if len(buffer) >= batch_size:
            collection.bulk_write(buffer, ordered=False)
            count += len(buffer)
            buffer = []

    # Insert leftover docs
    if buffer:
        collection.bulk_write(buffer, ordered=False)
        count += len(buffer)

    print(f"\nDone! Inserted {count} documents into MongoDB Atlas.")

# ---------------------------
# 4. Create Indexes (Important!)
# ---------------------------
def create_indexes():
    print("\n📌 Creating indexes for fast RAG querying...")
    col.create_index("asin")
    col.create_index("sentiment")
    col.create_index([("reviewText", "text")])  # full-text search
    print("Indexes created.\n")

# ---------------------------
# 5. Run Upload
# ---------------------------
if __name__ == "__main__":
    create_indexes()
    bulk_insert_jsonl("data/combined_reviews.jsonl", col)
