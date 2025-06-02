from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your models ONCE at startup
clf = joblib.load('best_sentiment_model.pkl')
le = joblib.load('label_encoder.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# If you have a preprocessing function, import or define it here
def clean_text(text):
    # Example: lowercase and strip (replace with your actual logic)
    return text.lower().strip()

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
