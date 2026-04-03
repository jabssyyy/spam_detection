"""
PHASE 6: FastAPI Backend
Serves spam predictions via REST API.
Run with: uvicorn api.main:app --reload
"""

import os, sys, time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_to_string
from src.vectorizer    import transform_texts

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
FRONTEND_DIR  = os.path.join(BASE_DIR, "frontend")

MODEL_FILES = {
    "naive_bayes"         : os.path.join(MODELS_DIR, "naive_bayes.pkl"),
    "logistic_regression" : os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    "svm"                 : os.path.join(MODELS_DIR, "svm.pkl"),
}
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
DEFAULT_MODEL   = "svm"

# ── App state (loaded once at startup) ────────────────────────────
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once when server starts. Release on shutdown."""
    print("[startup] Loading vectorizer and all models...")
    app_state["vectorizer"] = joblib.load(VECTORIZER_PATH)
    app_state["models"]     = {
        name: joblib.load(path) for name, path in MODEL_FILES.items()
    }
    app_state["started_at"] = time.time()
    print(f"[startup] Ready. Models loaded: {list(app_state['models'].keys())}")
    yield
    app_state.clear()
    print("[shutdown] Resources released.")


# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title       = "Spam Detection API",
    description = "End-to-end SMS spam classifier — ML learning project Phase 6",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS: allow requests from ANY origin (needed for browser-based frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    text  : str = Field(..., min_length=1,
                        example="Congratulations! You have won a FREE prize. Call NOW!")
    model : str = Field(default=DEFAULT_MODEL, example="svm")


class StepDetail(BaseModel):
    raw_text     : str
    cleaned_text : str
    token_count  : int


class PredictionResponse(BaseModel):
    prediction : str    # "spam" or "ham"
    label      : int    # 1=spam, 0=ham
    confidence : float  # probability of predicted class
    spam_prob  : float  # P(spam) — always explicit
    ham_prob   : float  # P(ham)  — always explicit
    model_used : str
    pipeline   : StepDetail


class BatchPredictionRequest(BaseModel):
    texts : list[str] = Field(..., min_length=1)
    model : str = Field(default=DEFAULT_MODEL)


class BatchPredictionResponse(BaseModel):
    results    : list[dict]
    model_used : str
    count      : int


class ModelInfoResponse(BaseModel):
    available_models : list[str]
    default_model    : str
    uptime_seconds   : float


# ══════════════════════════════════════════════════════════════════
# CORE PREDICTION LOGIC (shared by all endpoints)
# ══════════════════════════════════════════════════════════════════
def _run_prediction(text: str, model_name: str) -> PredictionResponse:
    """Preprocess → vectorize → predict → wrap response."""
    if model_name not in app_state["models"]:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unknown model '{model_name}'. "
                          f"Choose from: {list(app_state['models'].keys())}",
        )
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    cleaned    = preprocess_to_string(text)
    vector     = transform_texts([cleaned], app_state["vectorizer"])
    model      = app_state["models"][model_name]
    label      = int(model.predict(vector)[0])
    proba      = model.predict_proba(vector)[0]    # [P(ham), P(spam)]
    spam_prob  = round(float(proba[1]), 4)
    ham_prob   = round(float(proba[0]), 4)
    confidence = round(float(max(proba)), 4)

    return PredictionResponse(
        prediction = "spam" if label == 1 else "ham",
        label      = label,
        confidence = confidence,
        spam_prob  = spam_prob,
        ham_prob   = ham_prob,
        model_used = model_name,
        pipeline   = StepDetail(
            raw_text     = text,
            cleaned_text = cleaned,
            token_count  = len(cleaned.split()),
        ),
    )


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    """Returns server status and loaded model names."""
    return {
        "status"         : "healthy",
        "models_loaded"  : list(app_state.get("models", {}).keys()),
        "default_model"  : DEFAULT_MODEL,
        "uptime_seconds" : round(time.time() - app_state.get("started_at", time.time()), 1),
    }


@app.get("/app", tags=["Frontend"], include_in_schema=False)
def serve_ui():
    """Serve the frontend UI directly from FastAPI."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/models", response_model=ModelInfoResponse, tags=["Info"])
def model_info():
    """List all available models and which one is the default."""
    return ModelInfoResponse(
        available_models = list(app_state["models"].keys()),
        default_model    = DEFAULT_MODEL,
        uptime_seconds   = round(time.time() - app_state["started_at"], 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Classify a single SMS message as spam or ham.

    - **text**: raw message text (uncleaned — the API handles preprocessing)
    - **model**: `naive_bayes` | `logistic_regression` | `svm` (default: svm)
    """
    return _run_prediction(request.text, request.model)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Classify multiple messages in a single API call."""
    results = []
    for text in request.texts:
        try:
            r = _run_prediction(text, request.model)
            results.append({
                "text"       : text[:80],
                "prediction" : r.prediction,
                "confidence" : r.confidence,
                "spam_prob"  : r.spam_prob,
                "ham_prob"   : r.ham_prob,
            })
        except Exception as e:
            results.append({"text": text[:80], "error": str(e)})

    return BatchPredictionResponse(
        results    = results,
        model_used = request.model,
        count      = len(results),
    )


@app.get("/predict/compare", tags=["Prediction"])
def compare_all_models(text: str):
    """
    Run the same message through all 3 models simultaneously.
    Great for understanding how each model differs in confidence.
    """
    comparison = {}
    for model_name in app_state["models"]:
        r = _run_prediction(text, model_name)
        comparison[model_name] = {
            "prediction" : r.prediction,
            "spam_prob"  : r.spam_prob,
            "ham_prob"   : r.ham_prob,
            "confidence" : r.confidence,
        }
    return {"text": text, "comparison": comparison}


# Mount static files LAST (so API routes take priority)
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
