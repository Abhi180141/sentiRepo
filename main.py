from fastapi import FastAPI
from pydantic import BaseModel
import joblib

import nltk
from nltk.corpus import stopwords
import re

# Load your models ONCE at startup
clf = joblib.load('best_sentiment_model.pkl')
le = joblib.load('label_encoder.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


nltk.download('stopwords')

# Negation words to keep
negation_words = {
    "not", "no", "nor", "don't", "didn't", "doesn't", "hadn't", "hasn't", "haven't",
    "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't",
    "weren't", "won't", "wouldn't", "cannot", "can't", "couldn't"
}

stop_words = set(stopwords.words('english')) - negation_words

def handle_negations(text):
    # Convert contractions and negations into joined forms
    text = re.sub(r"n't", " not", text)
    # It searches for common negation words followed by a word, and joins them with an underscore.
    text = re.sub(r"\b(not|no|never|cannot|can't|won't|don't|doesn't|didn't|isn't|wasn't|shouldn't|wouldn't|couldn't|haven't|hasn't|hadn't|mustn't|needn't|mightn't|shan't|nor)\s+(\w+)", r"\1_\2", text)
    return text

# # If you have a preprocessing function, import or define it here
def clean_text(text):
    text = text.lower()
    text = handle_negations(text)
    text = re.sub(r"[^a-z_\s]", "", text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Request schema
class ReviewRequest(BaseModel):
    review: str

# Response schema
class SentimentResponse(BaseModel):
    sentiment: str

app = FastAPI()

@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(req: ReviewRequest):
    review_clean = clean_text(req.review)
    review_vec = vectorizer.transform([review_clean])
    prediction = clf.predict(review_vec)
    sentiment = le.inverse_transform(prediction)[0]
    return {"sentiment": sentiment}
