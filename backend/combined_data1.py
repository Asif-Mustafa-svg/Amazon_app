import json
from tqdm import tqdm

# ---------- PATHS ----------
REVIEWS_FILE = "data/Industrial_and_Scientific.jsonl"
META_FILE = "data/meta_Industrial_and_Scientific.jsonl"
PROCESSED_FILE = "data/processed_reviews.jsonl"
OUTPUT_FILE = "data/final_combined_440931.jsonl"

# ---------- HELPERS ----------

def load_jsonl(path):
    """Stream JSONL file line by line."""
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_meta(path):
    """Map asin -> meta information."""
    meta_map = {}
    for item in load_jsonl(path):
        asin = item.get("parent_asin") or item.get("asin")
        if asin:
            meta_map[asin] = {
                "product_title": item.get("title"),
                "brand": item.get("details", {}).get("Brand"),
                "category": item.get("main_category"),
            }
    return meta_map

def load_processed(path):
    """Map asin -> processed sentiment results."""
    proc_map = {}
    for item in load_jsonl(path):
        asin = item.get("asin")
        if asin:
            proc_map[asin] = item
    return proc_map

# ---------- MAIN COMBINER ----------

def main():
    print("Loading META...")
    meta_map = load_meta(META_FILE)
    print("Meta count:", len(meta_map))

    print("Loading PROCESSED...")
    processed_map = load_processed(PROCESSED_FILE)
    print("Processed count:", len(processed_map))

    print("Combining all files...")
    count = 0

    with open(OUTPUT_FILE, "w") as out:
        for review in tqdm(load_jsonl(REVIEWS_FILE), total=440931):
            
            asin = review.get("asin")
            processed = processed_map.get(asin, {})  # sentiment etc.
            meta = meta_map.get(asin, {})           # title, brand, category
            
            combined = {
                # IDs
                "asin": asin,
                "parent_asin": review.get("parent_asin"),

                # Raw review data
                "rating": review.get("rating"),
                "title": review.get("title"),
                "review_text": review.get("text"),

                # Meta
                "product_title": meta.get("product_title"),
                "brand": meta.get("brand"),
                "category": meta.get("category"),

                # Processed (sentiment/NER)
                "clean_text": processed.get("clean_text", ""),
                "vader_compound": processed.get("vader_compound"),
                "vader_sentiment": processed.get("vader_sentiment"),
                "bert_sentiment": processed.get("bert_sentiment"),
                "bert_score": processed.get("bert_score"),
                "product_mentions": processed.get("product_mentions", []),
            }

            out.write(json.dumps(combined) + "\n")
            count += 1

            if count >= 440931:
                break

    print("DONE! Final combined file saved as:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
