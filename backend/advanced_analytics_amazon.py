import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import CountVectorizer


FILE = "data/final_combined_440931.jsonl"

print("Loading dataset...")
df = pd.read_json(FILE, lines=True)

print(f"Loaded {len(df)} reviews")

df["review_text"] = df["review_text"].astype(str)
df["category"] = df["category"].fillna("Unknown")

os.makedirs("plots", exist_ok=True)
os.makedirs("analytics", exist_ok=True)

# -------------------------------------
# 1. VADER sentiment distribution
# -------------------------------------
plt.figure(figsize=(6,4))
df["vader_sentiment"].value_counts().plot(kind="bar", color="skyblue")
plt.title("VADER Sentiment Distribution")
plt.savefig("plots/vader_sentiment_distribution.png")
plt.close()

# -------------------------------------
# 2. BERT sentiment distribution
# -------------------------------------
plt.figure(figsize=(6,4))
df["bert_sentiment"].value_counts().plot(kind="bar", color="orange")
plt.title("BERT Sentiment Distribution")
plt.savefig("plots/bert_sentiment_distribution.png")
plt.close()

# -------------------------------------
# 3. Category stats (FAST)
# -------------------------------------
print("Calculating category stats...")

category_stats = (
    df.groupby("category")
    .agg(
        avg_rating=("rating", "mean"),
        avg_vader=("vader_compound", "mean"),
        review_count=("asin", "count")
    )
    .sort_values("review_count", ascending=False)
)

category_stats.to_csv("analytics/category_stats.csv")

print("\nTop 10 categories by review count:")
print(category_stats.head(10))

# -------------------------------------
# 4. FAST extraction of negative keywords
# -------------------------------------
print("Extracting negative review keywords...")

neg_texts = df.loc[df["rating"] <= 2, "review_text"].head(5000)  # limit for speed


vectorizer = CountVectorizer(stop_words="english", max_features=50)
X = vectorizer.fit_transform(neg_texts)
keywords = vectorizer.get_feature_names_out()

pd.DataFrame({"keyword": keywords}).to_csv("analytics/negative_keywords.csv", index=False)

print("\nTop Negative Keywords:")
print(keywords[:20])

print("\nAnalytics Completed Successfully!")
