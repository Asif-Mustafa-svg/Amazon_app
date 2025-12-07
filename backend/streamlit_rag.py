# streamlit_rag_app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss
import json
import re
from langchain_ollama import ChatOllama

# ----------------- CONFIG -----------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data", "final_combined_440931.jsonl")  # FIXED
MONGODB_URI = os.getenv("MONGODB_URI")  # required
MONGO_DB = os.getenv("MONGO_DB", "amazon_reviews")
MONGO_COL = os.getenv("MONGO_COL", "combined_reviews")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_PATH = os.path.join(BASE_DIR, "../data", "faiss_index.pkl")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

st.set_page_config(layout="wide", page_title="Amazon RAG + Analytics")

# ----------------- LLM -----------------
@st.cache_resource
def get_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.0)

llm = get_llm()

def llm_text(out):
    return out.content if hasattr(out, "content") else str(out)

# ----------------- Load dataset -----------------
@st.cache_data
def load_local_df():
    if os.path.exists(DATA_PATH):
        return pd.read_json(DATA_PATH, lines=True)
    return pd.DataFrame()

df = load_local_df()

# ----------------- Mongo connection -----------------
@st.cache_resource
def get_mongo():
    if MONGODB_URI is None or MONGODB_URI.strip() == "":
        return None
    client = MongoClient(MONGODB_URI)
    return client[MONGO_DB][MONGO_COL]

col = get_mongo()

# ----------------- Embeddings -----------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = get_embedder()

# ----------------- FAISS load -----------------
def load_faiss():
    if os.path.exists(FAISS_PATH):
        with open(FAISS_PATH, "rb") as f:
            data = pickle.load(f)
            return data["index"], data["embeddings"], data["meta"]
    return None, None, None

faiss_index, faiss_embeddings, faiss_meta = load_faiss()

# ----------------- Helpers: atlas vector check -----------------
def has_atlas_embedding(collection):
    if collection is None:   # FIXED
        return False
    try:
        return collection.find_one({"embedding": {"$exists": True}}) is not None
    except:
        return False

USE_ATLAS = has_atlas_embedding(col)

# ----------------- Retrieval -----------------
def retrieve_from_atlas(query, k=5):
    q_vec = embedder.encode([query])[0].tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": q_vec,
                "path": "embedding",
                "k": k
            }
        }
    ]
    docs = list(col.aggregate(pipeline))
    return [
        {"asin": d.get("asin"), "review_text": d.get("review_text", ""), "rating": d.get("rating")}
        for d in docs
    ]

def retrieve_local_faiss(query, k=5):
    if faiss_index is None:
        return []
    q_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = faiss_index.search(q_vec, k)
    return [faiss_meta[idx] for idx in I[0] if idx < len(faiss_meta)]

def retrieve(query, k=5):
    if USE_ATLAS and col is not None:
        try:
            return retrieve_from_atlas(query, k)
        except Exception as e:
            st.warning(f"Atlas vector search error: {e}. Falling back to local.")
            return retrieve_local_faiss(query, k)
    else:
        return retrieve_local_faiss(query, k)

# ----------------- Filter LLM -----------------
PROMPT_FILTER = """
You are a MongoDB filter generator for an Amazon reviews database.

User query: "{question}"

Return ONLY a JSON object with keys:
- asin: ASIN string or null
- sentiment: "positive", "negative", "neutral", or null
- keyword: the single most important keyword or null

STRICT JSON ONLY. No explanation.
"""

def make_filter(question: str):
    prompt = PROMPT_FILTER.format(question=question)
    raw = llm.invoke(prompt)
    raw_text = llm_text(raw)

    m = re.search(r'\{.*\}', raw_text, flags=re.S)
    if not m:
        try:
            return json.loads(raw_text)
        except:
            return {"asin": None, "sentiment": None, "keyword": None, "raw": raw_text}
    try:
        return json.loads(m.group(0))
    except:
        return {"asin": None, "sentiment": None, "keyword": None, "raw": raw_text}

# ----------------- RAG Answer -----------------
RAG_PROMPT = """
You are an assistant. Use ONLY the provided review context.

If answer is not present, reply: "I don't know."

Context:
{context}

Question:
{question}

Answer concisely.
"""

def rag_answer(question: str, k=5):
    hits = retrieve(question, k=k)
    if not hits:
        return "No relevant documents found.", []

    context = "\n\n".join(
        [f"ASIN: {h.get('asin')} | Rating: {h.get('rating')}\n{h.get('review_text')}" for h in hits]
    )

    prompt = RAG_PROMPT.format(context=context, question=question)
    out = llm.invoke(prompt)
    return llm_text(out), hits

# ----------------- Streamlit UI -----------------
st.title("Amazon Reviews — Analytics + RAG Chat")

left, right = st.columns([2, 1])

# --- Analytics ---
with left:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Unique ASINs", df["asin"].nunique() if not df.empty else 0)
    col3.metric("Avg Rating", round(df["rating"].mean(), 2) if not df.empty else "N/A")

    st.subheader("Rating Distribution")
    if not df.empty:
        fig, ax = plt.subplots()
        df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.subheader("Top Categories")
    if not df.empty:
        st.bar_chart(df["category"].fillna("Unknown").value_counts().head(20))

    st.subheader("Search & Retrieve Reviews")
    q = st.text_input("Your search or query:", value="mask quality")
    k = st.slider("k (retrieved docs)", 1, 10, 5)

    if st.button("Retrieve"):
        hits = retrieve(q, k=k)
        if not hits:
            st.write("No hits.")
        else:
            for h in hits:
                st.markdown(f"**ASIN:** {h.get('asin')} — Rating: {h.get('rating')}")
                st.write(h.get("review_text"))

# --- Chat / RAG ---
with right:
    st.header("RAG Chatbot")

    user_q = st.text_area("Ask about products or reviews:", height=150)

    if st.button("Ask"):
        if not user_q.strip():
            st.warning("Enter a question")
        else:
            flt = make_filter(user_q)
            st.write("**LLM Filter Output**")
            st.json(flt)

            # Build filter
            query_filter = {}
            if flt.get("asin"):
                query_filter["asin"] = flt["asin"]
            if flt.get("keyword"):
                query_filter["review_text"] = {"$regex": flt["keyword"], "$options": "i"}
            if flt.get("sentiment"):
                query_filter["bert_sentiment"] = {"$regex": flt["sentiment"], "$options": "i"}

            context_docs = []
            if col is not None and query_filter:
                try:
                    docs = list(col.find(query_filter).limit(200))
                    context_docs = docs[:k]
                except Exception as e:
                    st.warning(f"Mongo filter error: {e}")

            if not context_docs:
                context_docs = retrieve(user_q, k=k)

            if not context_docs:
                st.info("I don't know.")
                st.write("No documents found.")
            else:
                ans, used = rag_answer(user_q, k=k)
                st.subheader("Answer")
                st.write(ans)

                st.subheader("Context Used")
                for d in used:
                    st.markdown(f"**ASIN:** {d.get('asin')} — Rating: {d.get('rating')}")
                    st.write(d.get('review_text'))

# --- Debug Sidebar ---
st.sidebar.header("Debug")
st.sidebar.write(f"Using Atlas Vector Search: {USE_ATLAS}")
st.sidebar.write(f"Ollama Model: {OLLAMA_MODEL}")
st.sidebar.write(f"FAISS index present: {os.path.exists(FAISS_PATH)}")
