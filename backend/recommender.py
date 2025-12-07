import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

DATA_FILE = "data/final_combined_440931.jsonl"

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------

print("Loading dataset...")
df = pd.read_json(DATA_FILE, lines=True)

# Create artificial user IDs since dataset has none
df["reviewerID"] = df.index.astype(str)

# -----------------------------------------------------------
# 2. CONTENT-BASED MODEL (TF-IDF)
# -----------------------------------------------------------

print("Preparing product-level aggregated text...")

product_df = (
    df.groupby("parent_asin")
      .agg(
          review_text=("review_text", lambda x: " ".join(x.astype(str))),
          avg_rating=("rating", "mean"),
          review_count=("asin", "count")
      )
)

asin_list = product_df.index.tolist()

print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.85)
tfidf_matrix = tfidf.fit_transform(product_df["review_text"])

print("Building NearestNeighbors index...")
nn = NearestNeighbors(metric="cosine", algorithm="brute")
nn.fit(tfidf_matrix)

distances, neighbors = nn.kneighbors(tfidf_matrix, n_neighbors=21)

def recommend_cb(asin, top_n=10):
    """Content-Based Recommendation"""
    if asin not in product_df.index:
        return f"ASIN '{asin}' not found."

    idx = product_df.index.get_loc(asin)
    neighbor_idxs = neighbors[idx][1: top_n + 1]
    sim_scores = 1 - distances[idx][1: top_n + 1]

    rec_asins = [asin_list[i] for i in neighbor_idxs]

    return pd.DataFrame({
        "recommended_asin": rec_asins,
        "cb_score": sim_scores,
        "avg_rating": product_df.loc[rec_asins]["avg_rating"].values,
        "review_count": product_df.loc[rec_asins]["review_count"].values
    })


# -----------------------------------------------------------
# 3. SPARSE COLLABORATIVE FILTERING (SAFE)
# -----------------------------------------------------------

print("Encoding users and products for sparse matrix...")

user_enc = LabelEncoder()
item_enc = LabelEncoder()

df["user_enc"] = user_enc.fit_transform(df["reviewerID"])
df["item_enc"] = item_enc.fit_transform(df["parent_asin"])

num_users = df["user_enc"].nunique()
num_items = df["item_enc"].nunique()

print(f"Users: {num_users}, Items: {num_items}")

print("Building sparse user–item rating matrix...")

ratings_sparse = csr_matrix(
    (df["rating"], (df["user_enc"], df["item_enc"])),
    shape=(num_users, num_items)
)

print("Computing sparse item-item similarity...")
item_similarity_sparse = cosine_similarity(
    ratings_sparse.T,
    dense_output=False  # important! prevents huge memory usage
)

def recommend_cf(asin, top_n=10):
    """Collaborative Filtering using Sparse Matrix"""
    if asin not in product_df.index:
        return f"ASIN '{asin}' not found."

    item_id = item_enc.transform([asin])[0]

    # get sparse similarity vector
    sim_vec = item_similarity_sparse[item_id].toarray().flatten()

    top_idx = sim_vec.argsort()[::-1][1 : top_n + 1]
    rec_asins = item_enc.inverse_transform(top_idx)
    sim_scores = sim_vec[top_idx]

    return pd.DataFrame({
        "recommended_asin": rec_asins,
        "cf_score": sim_scores,
        "avg_rating": product_df.loc[rec_asins]["avg_rating"].values,
        "review_count": product_df.loc[rec_asins]["review_count"].values
    })


# -----------------------------------------------------------
# 4. HYBRID RECOMMENDER
# -----------------------------------------------------------

def recommend_hybrid(asin, top_n=10, alpha=0.5):
    cb = recommend_cb(asin, top_n=50)
    cf = recommend_cf(asin, top_n=50)

    if isinstance(cb, str) or isinstance(cf, str):
        return "ASIN not found in one of the models."

    merged = cb.merge(
        cf[["recommended_asin", "cf_score"]],
        on="recommended_asin",
        how="inner"
    )

    merged["hybrid_score"] = alpha * merged["cb_score"] + (1 - alpha) * merged["cf_score"]
    return merged.sort_values("hybrid_score", ascending=False).head(top_n)


# -----------------------------------------------------------
# 5. INTERACTIVE INPUT
# -----------------------------------------------------------

print("\nRecommender READY.")

while True:
    asin = input("\nEnter ASIN (or 'quit'): ").strip()
    if asin.lower() == "quit":
        break

    print("Choose method:\n1 = Content-Based\n2 = Collaborative Filtering\n3 = Hybrid")
    choice = input("Enter 1/2/3: ").strip()

    if choice == "1":
        print(recommend_cb(asin))
    elif choice == "2":
        print(recommend_cf(asin))
    elif choice == "3":
        alpha = float(input("Alpha (0-1): ") or 0.5)
        print(recommend_hybrid(asin, alpha=alpha))
    else:
        print("Invalid choice.")
