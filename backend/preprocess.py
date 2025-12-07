import json
from tqdm import tqdm
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
nltk.download("punkt")

# --------------------------------------------
# LOAD MODELS (lightweight for M1)
# --------------------------------------------

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

print("Loading VADER...")
vader = SentimentIntensityAnalyzer()

print("Loading DistilBERT sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# --------------------------------------------
# CLEAN TEXT
# --------------------------------------------

def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    return text

# --------------------------------------------
# PROCESS ONE REVIEW
# --------------------------------------------

def process_review(text):
    text = clean_text(text)

    # Sentiment
    vader_result = vader.polarity_scores(text)
    hf_result = sentiment_pipeline(text)[0]

    # spaCy NER
    doc = nlp(text)
    products = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]

    return {
        "clean_text": text,
        "vader_compound": vader_result["compound"],
        "vader_sentiment": (
            "positive" if vader_result["compound"] > 0.05
            else "negative" if vader_result["compound"] < -0.05
            else "neutral"
        ),
        "bert_sentiment": hf_result["label"],
        "bert_score": hf_result["score"],
        "product_mentions": products
    }

# --------------------------------------------
# LOAD JSONL DATA
# --------------------------------------------

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

# --------------------------------------------
# PIPELINE
# --------------------------------------------

REVIEWS = "data/Industrial_and_Scientific.jsonl"
META = "data/meta_Industrial_and_Scientific.jsonl"

output = []

print("Processing reviews...")
for item in tqdm(load_jsonl(REVIEWS)):
    text = item.get("reviewText", "")
    result = process_review(text)
    result["asin"] = item.get("asin")
    result["rating"] = item.get("overall")
    output.append(result)

with open("processed.jsonl", "w") as f:
    for row in output:
        f.write(json.dumps(row) + "\n")

print("Done! Saved to processed.jsonl")

import json
with open("processed_reviews.jsonl", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")
