# ═══════════════════════════════════════════════════════════════
#  model.py  —  SentimentEdge
#  Loads FinBERT from local disk only. No internet after setup.
#  Fix: local_files_only passed to model/tokenizer load, not pipeline
# ═══════════════════════════════════════════════════════════════

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL  = os.path.join(BASE_DIR, "models", "finbert")
REMOTE_MODEL = "ProsusAI/finbert"

_finbert_pipeline = None


def _get_model_path() -> tuple:
    """
    Returns (model_path, is_local).
    Checks local disk first, falls back to remote.
    """
    config_path = os.path.join(LOCAL_MODEL, "config.json")

    if os.path.exists(config_path):
        return LOCAL_MODEL, True
    else:
        print("\n" + "!"*55)
        print("  WARNING: Local FinBERT not found.")
        print(f"  Expected at: {LOCAL_MODEL}")
        print("  Falling back to HuggingFace (needs internet).")
        print("  To go fully offline, run once:")
        print("    python download_models.py")
        print("!"*55 + "\n")
        return REMOTE_MODEL, False


def load_finbert() -> None:
    """Load FinBERT once on server start."""
    global _finbert_pipeline

    if _finbert_pipeline is not None:
        return

    model_path, is_local = _get_model_path()
    source = "local disk ✓ (offline)" if is_local else "HuggingFace (online)"

    print(f"[model] Loading FinBERT from {source}")
    print(f"[model] Path: {model_path}")

    device = 0 if torch.cuda.is_available() else -1
    print(f"[model] Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    # ── Load tokenizer and model separately with local_files_only ──
    # This is the correct place for local_files_only — NOT in pipeline()
    kwargs = {"local_files_only": is_local}

    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    model_path, **kwargs)

    # ── Build pipeline from already-loaded objects ──
    # Do NOT pass local_files_only here — pipeline() doesn't accept it
    _finbert_pipeline = pipeline(
        "sentiment-analysis",
        model     = model,
        tokenizer = tokenizer,
        device    = device,
        truncation= True,
        max_length= 512,
    )

    print("[model] FinBERT ready.\n")


def _ensure_loaded():
    if _finbert_pipeline is None:
        load_finbert()


# ════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════

def classify_text(text: str) -> dict:
    """Classify a single text. Returns label, score, polarity."""
    _ensure_loaded()

    result    = _finbert_pipeline(str(text).strip()[:512])[0]
    label     = result["label"].lower()
    score     = round(result["score"], 4)
    score_pct = round(score * 100, 1)
    polarity  = score if label == "positive" else (-score if label == "negative" else 0.0)

    return {
        "label":     label,
        "score":     score,
        "score_pct": score_pct,
        "polarity":  round(polarity, 4),
    }


def classify_batch(texts: list, batch_size: int = 32) -> list:
    """Classify a list of texts efficiently."""
    _ensure_loaded()

    cleaned = [str(t).strip()[:512] for t in texts]
    results = _finbert_pipeline(cleaned, batch_size=batch_size,
                                truncation=True, max_length=512)
    output  = []
    for r in results:
        label     = r["label"].lower()
        score     = round(r["score"], 4)
        score_pct = round(score * 100, 1)
        polarity  = score if label == "positive" else (-score if label == "negative" else 0.0)
        output.append({
            "label":     label,
            "score":     score,
            "score_pct": score_pct,
            "polarity":  round(polarity, 4),
        })
    return output


def get_model_status() -> dict:
    """Return model status for /api/status route."""
    config_path = os.path.join(LOCAL_MODEL, "config.json")
    is_local    = os.path.exists(config_path)
    size_mb     = 0.0
    if is_local:
        for f in os.listdir(LOCAL_MODEL):
            fp = os.path.join(LOCAL_MODEL, f)
            if os.path.isfile(fp):
                size_mb += os.path.getsize(fp) / 1_048_576
    return {
        "loaded":        _finbert_pipeline is not None,
        "offline_ready": is_local,
       "model_path": LOCAL_MODEL if is_local else "HuggingFace Hub",
        "model_size_mb": round(size_mb, 1),
        "device":        "cuda" if torch.cuda.is_available() else "cpu",
    }