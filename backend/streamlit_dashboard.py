import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

# --------------------------
# SAFE JSONL Loader
# --------------------------
@st.cache_data
def load_data():
    rows = []
    bad = 0

    with open("data/final_combined_440931.jsonl", "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
                continue  # skip bad line

    df_loaded = pd.DataFrame(rows)
    print("Loaded rows:", len(df_loaded))
    print("Skipped bad rows:", bad)
    return df_loaded


df = load_data()

# --------------------------
# Dashboard Title
# --------------------------
st.title(" Amazon Reviews Analytics Dashboard")
st.markdown("Interactive dashboard for exploring Amazon review dataset.")

# --------------------------
# Overview Stats
# --------------------------
st.subheader("Overview Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Unique ASINs", f"{df['asin'].nunique():,}")
col3.metric("Average Rating", round(df["rating"].mean(), 2))

# --------------------------
# Rating Distribution
# --------------------------
st.subheader("Rating Distribution")
fig, ax = plt.subplots()
df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
st.pyplot(fig)

# --------------------------
# Sentiment Distribution
# --------------------------
st.subheader("BERT Sentiment Distribution")
fig, ax = plt.subplots()
df["bert_sentiment"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

# --------------------------
# Top Categories
# --------------------------
st.subheader("Top Categories")
top_cats = df["category"].value_counts().head(20)
st.bar_chart(top_cats)

# --------------------------
# Top Brands
# --------------------------
st.subheader("Top Brands")
top_brands = df["brand"].value_counts().head(20)
st.bar_chart(top_brands)

# --------------------------
# ASIN Explorer
# --------------------------
st.subheader("ASIN-Level Review Explorer")

asin_list = df["asin"].dropna().unique().tolist()
selected_asin = st.selectbox("Choose ASIN:", asin_list)

asin_df = df[df["asin"] == selected_asin]

st.write(f"### {len(asin_df)} reviews found for **{selected_asin}**")

# Rating distribution for ASIN
fig, ax = plt.subplots()
asin_df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
st.pyplot(fig)

# --------------------------
# WordCloud Section
# --------------------------
st.subheader("Wordcloud of Reviews")

text = " ".join(asin_df["review_text"].astype(str).tolist())

if len(text) > 10:
    wc = WordCloud(width=1200, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("Not enough text for wordcloud.")

# --------------------------
# Show Sample Reviews
# --------------------------
st.subheader("Sample Reviews")
st.write(asin_df[["rating", "title", "review_text"]].head(10))
