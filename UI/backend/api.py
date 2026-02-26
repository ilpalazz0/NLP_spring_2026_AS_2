import os
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import tokenize, make_features


# Pre-computed n-gram results 
NGRAM_STATS = {
    "n_tokens": 199725,
    "vocab_size": 18723,
    "n_unigrams": 18723,
    "n_bigrams": 120982,
    "n_trigrams": 175442
}

NGRAM_PERPLEXITY = {
    "none": {
        "unigram": 330.1563,
        "bigram": 19.8757,
        "trigram": 5.3282
    },
    "laplace": {
        "unigram": 341.3685,
        "bigram": 108.5696,
        "trigram": 32.6923
    },
    "k": {
        "unigram": 335.478,
        "bigram": 84.5039,
        "trigram": 22.1084
    },
    "interpolation": {
        "unigram": 330.1563,
        "bigram": 19.8757,
        "trigram": 5.3282
    },
    "backoff": {
        "unigram": 330.1563,
        "bigram": 19.8757,
        "trigram": 5.3282
    },
    "kneser_ney": {
        "unigram": 330.1563,
        "bigram": 20.3635,
        "trigram": 4.4173
    }
}

# App setup
app = FastAPI(title="AZ Sentiment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = "saved"

def _load(filename):
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model file '{path}' not found. Run  python train.py  first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    vocab   = _load("vocab.pkl")
    nb      = _load("nb.pkl")
    bnb     = _load("bnb.pkl")
    lr      = _load("lr.pkl")
    metrics = _load("metrics.pkl")
    print("✓ All models loaded successfully.")
except RuntimeError as e:
    print(f"✗ {e}")
    vocab = nb = bnb = lr = metrics = None




class TextInput(BaseModel):
    text: str


# Routes
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": vocab is not None}


@app.get("/metrics")
def get_metrics():
    """Return pre-computed evaluation metrics from training."""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")
    return metrics


@app.post("/predict")
def predict(input: TextInput):
    """
    Predict sentiment for a given text.
    Returns label + confidence (0–1 probability of positive) for each model.
    """
    if vocab is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train.py first.")

    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    tokens    = tokenize(text)
    feats_nb  = make_features(tokens, vocab, binary=False)
    feats_bnb = make_features(tokens, vocab, binary=True)

    def label(pred):
        return "positive" if pred == 1 else "negative"

    nb_prob  = nb.predict_proba(feats_nb)
    bnb_prob = bnb.predict_proba(feats_bnb)
    lr_prob  = lr.predict_proba(feats_nb)

    return {
        "tokens_found": len(tokens),
        "nb":  {"label": label(nb.predict(feats_nb)),   "confidence": round(nb_prob,  4)},
        "bnb": {"label": label(bnb.predict(feats_bnb)), "confidence": round(bnb_prob, 4)},
        "lr":  {"label": label(lr.predict(feats_nb)),   "confidence": round(lr_prob,  4)},
    }


# N-gram routes

VALID_MODES = {"none", "laplace", "k", "interpolation", "backoff", "kneser_ney"}


@app.get("/ngram/stats")
def ngram_stats_route():
    if NGRAM_STATS["n_tokens"] is None:
        raise HTTPException(status_code=503, detail="N-gram stats not yet populated. Run precompute_ngrams.py first.")
    return NGRAM_STATS


@app.get("/ngram/perplexity")
def ngram_perplexity(mode: str = "laplace"):
    if mode not in VALID_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid mode. Choose from: {', '.join(sorted(VALID_MODES))}")
    if NGRAM_PERPLEXITY["laplace"]["unigram"] is None:
        raise HTTPException(status_code=503, detail="Perplexity data not yet populated. Run precompute_ngrams.py first.")
    return {"mode": mode, **NGRAM_PERPLEXITY[mode]}