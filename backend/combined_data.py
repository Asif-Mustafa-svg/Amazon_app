import json
from tqdm import tqdm

# --------- FILE PATHS (change if needed) ----------
REVIEWS_FILE = "data/Industrial_and_Scientific.jsonl"
META_FILE = "data/meta_Industrial_and_Scientific.jsonl"
PROCESSED_FILE = "data/processed_reviews.jsonl"  # your earlier output
OUTPUT_FILE = "data/combined_reviews.jsonl"


def load_jsonl(path):
    """Stream JSONL file line by line."""
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_meta(path):
    """Load meta file into dict: asin -> meta-info."""
    meta_map = {}
    for item in load_jsonl(path):
        asin = item.get("asin")
        if asin:
            meta_map[asin] = {
                "title": item.get("title"),
                "brand": item.get("brand"),
                "category": item.get("category"),
            }
    return meta_map


def main():
    print("Loading metadata (meta_Industrial_and_Scientific)...")
    meta_map = load_meta(META_FILE)
    print(f"Meta entries: {len(meta_map)}")

    print("Opening reviews and processed files...")
    rev_iter = load_jsonl(REVIEWS_FILE)
    proc_iter = load_jsonl(PROCESSED_FILE)

    with open(OUTPUT_FILE, "w") as out_f:
        for rev, proc in tqdm(zip(rev_iter, proc_iter), desc="Combining", unit="review"):
            asin = rev.get("asin")

            meta = meta_map.get(asin, {})
            combined = {
                # IDs
                "asin": asin,

                # from original review file
                "reviewText": rev.get("reviewText", ""),
                "summary": rev.get("summary", ""),
                "rating": rev.get("overall", None),

                # from meta
                "product_title": meta.get("title"),
                "brand": meta.get("brand"),
                "category": meta.get("category"),

                # from processed_reviews.jsonl (your earlier script)
                "clean_text": proc.get("clean_text", ""),
                "vader_compound": proc.get("vader_compound"),
                "vader_sentiment": proc.get("vader_sentiment"),
                "bert_sentiment": proc.get("bert_sentiment"),
                "bert_score": proc.get("bert_score"),
                "product_mentions": proc.get("product_mentions", []),
            }

            out_f.write(json.dumps(combined) + "\n")

    print(f"Done! Wrote combined data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
