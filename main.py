from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle, re
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="PhishGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ───────────────────────────────────────────────
print("Loading models...")

url_model   = tf.keras.models.load_model('saved_models/url_ann_model.h5')
email_model = tf.keras.models.load_model('saved_models/email_cnn_model.h5')
ceas_model  = tf.keras.models.load_model('saved_models/ceas_cnn_model.h5')

scaler        = pickle.load(open('saved_models/url_scaler.pkl',        'rb'))
selector      = pickle.load(open('saved_models/url_selector.pkl',      'rb'))
selected_cols = pickle.load(open('saved_models/url_selected_cols.pkl', 'rb'))
email_tok     = pickle.load(open('saved_models/email_tokenizer.pkl',   'rb'))
ceas_tok      = pickle.load(open('saved_models/ceas_tokenizer.pkl',    'rb'))

print("✅ All models loaded!")

MAX_LEN = 100

# ── Request Schemas ───────────────────────────────────────────
class URLRequest(BaseModel):
    url: str

class EmailRequest(BaseModel):
    text: str
    dataset: str = "email"

# ── Helper: Clean Text ────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "✅ PhishGuard API is running!"}

@app.post("/predict/url")
def predict_url(req: URLRequest):
    try:
        url = req.url.lower()
        domain = url.split('/')[2] if '//' in url else url
        score = 0

        # Negative signals (phishing)
        if not req.url.startswith('https'):          score += 20
        if re.search(r'\d+\.\d+\.\d+\.\d+', url):   score += 40
        if '@' in url:                               score += 30
        if url.count('-') > 3:                       score += 20
        if domain.count('.') > 3:                    score += 20
        if len(url) > 75:                            score += 15
        if url.count('=') + url.count('&') > 3:      score += 15
        if '%' in url:                               score += 10
        if '//' in url[8:]:                          score += 20

        # Positive signals (legitimate)
        if req.url.startswith('https'):              score -= 15
        if domain.count('.') == 1:                   score -= 10

        # Brand impersonation
        brands = ['paypal','google','apple','amazon',
                  'microsoft','bank','ebay','netflix']
        domain_name = domain.split('.')[0]
        for brand in brands:
            if brand in url and brand not in domain_name:
                score += 40
                break

        # Suspicious keywords
        keywords = ['login','verify','secure','account','update',
                    'confirm','banking','password','suspended',
                    'urgent','signin','webscr','validate','click']
        score += sum(1 for w in keywords if w in url) * 15

        prob = max(0.01, min(score / 150, 0.99))

        print(f"URL: {req.url} | score: {score} | prob: {prob:.4f}")

        return {
            "url":         req.url,
            "prediction":  "phishing" if prob > 0.4 else "legitimate",
            "confidence":  round((prob if prob > 0.4 else 1 - prob) * 100, 2),
            "probability": round(prob, 4)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/email")
def predict_email(req: EmailRequest):
    try:
        cleaned = clean_text(req.text)
        tok     = ceas_tok   if req.dataset == "ceas" else email_tok
        model   = ceas_model if req.dataset == "ceas" else email_model

        seq    = tok.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN,
                               padding='post', truncating='post')
        prob   = float(model.predict(padded, verbose=0)[0][0])

        print(f"Email prob: {prob:.4f} | dataset: {req.dataset}")

        return {
            "prediction":   "phishing" if prob > 0.5 else "legitimate",
            "confidence":   round((prob if prob > 0.5 else 1 - prob) * 100, 2),
            "probability":  round(prob, 4),
            "dataset_used": req.dataset
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {
        "url_model":   "loaded" if url_model   else "missing",
        "email_model": "loaded" if email_model else "missing",
        "ceas_model":  "loaded" if ceas_model  else "missing",
    }