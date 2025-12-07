# ingest_and_index.py
import os
import json
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import faiss
import pickle

# -------- CONFIG ----------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data", "final_combined_440931.jsonl")  # FIXED
MONGODB_URI = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGO_DB", "amazon_reviews")
MONGO_COL = os.getenv("MONGO_COL", "combined_reviews")
BATCH = int(os.getenv("BATCH", 512))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_PATH = os.path.join(BASE_DIR, "../data", "faiss_index.pkl")
# --------------------------

if not MONGODB_URI:
    raise SystemExit("Set MONGODB_URI environment variable and re-run.")

client = MongoClient(MONGODB_URI)
col = client[MONGO_DB][MONGO_COL]

print("Loading embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# --------- 1) UPSERT EMBEDDINGS INTO MONGO -----------
buffer = []
ops = []
texts = []
metas = []
count = 0

print("Starting embedding + Mongo upsert...")
for doc in tqdm(stream_jsonl(DATA_PATH)):
    review_text = doc.get("review_text") or doc.get("clean_text") or ""
    if not review_text.strip():
        continue

    buffer.append((doc, review_text))
    if len(buffer) >= BATCH:
        batch_docs, batch_texts = zip(*buffer)
        embs = embedder.encode(list(batch_texts), convert_to_numpy=True)

        for d, emb in zip(batch_docs, embs):
            # ensure unique key
            d_id = d.get("_id", f"{d.get('asin')}-{count}")
            count += 1

            ops.append(UpdateOne(
                {"_id": d_id},
                {"$set": {
                    "asin": d.get("asin"),
                    "review_text": d.get("review_text"),
                    "rating": d.get("rating"),
                    "embedding": emb.tolist()
                }},
                upsert=True
            ))

            # For FAISS
            texts.append(d.get("review_text"))
            metas.append({
                "asin": d.get("asin"),
                "rating": d.get("rating"),
                "review_text": d.get("review_text")  # FIXED
            })

        col.bulk_write(ops, ordered=False)
        ops = []
        buffer = []

# leftover
if buffer:
    batch_docs, batch_texts = zip(*buffer)
    embs = embedder.encode(list(batch_texts), convert_to_numpy=True)

    for d, emb in zip(batch_docs, embs):
        d_id = d.get("_id", f"{d.get('asin')}-{count}")
        count += 1

        ops.append(UpdateOne(
            {"_id": d_id},
            {"$set": {
                "asin": d.get("asin"),
                "review_text": d.get("review_text"),
                "rating": d.get("rating"),
                "embedding": emb.tolist()
            }},
            upsert=True
        ))

        metas.append({
            "asin": d.get("asin"),
            "rating": d.get("rating"),
            "review_text": d.get("review_text")
        })

    col.bulk_write(ops, ordered=False)

print("Finished writing embeddings into Mongo.")

# --------- 2) BUILD LOCAL FAISS INDEX ----------
print("Building FAISS index...")

docs_cursor = col.find({"embedding": {"$exists": True}}, {"embedding": 1, "asin": 1, "rating": 1, "review_text": 1})
emb_list = []
meta_list = []

for d in tqdm(docs_cursor):
    emb = d.get("embedding")
    if emb:
        emb_list.append(np.array(emb, dtype=np.float32))
        meta_list.append({
            "asin": d.get("asin"),
            "rating": d.get("rating"),
            "review_text": d.get("review_text")
        })

if emb_list:
    X = np.vstack(emb_list)
    faiss.normalize_L2(X)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)

    with open(FAISS_PATH, "wb") as f:
        pickle.dump({"index": index, "embeddings": X, "meta": meta_list}, f)

    print(f"Saved FAISS index to {FAISS_PATH}")
else:
    print("No embeddings available for FAISS.")

print("Done.")
