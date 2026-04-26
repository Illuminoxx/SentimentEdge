# ═══════════════════════════════════════════════════════════════
#  predict.py  —  SentimentEdge
#
#  This is the prediction engine.
#  Flow:
#    1. Receive tweet text (and optional extras)
#    2. FinBERT → sentiment label + polarity
#    3. Build the SAME feature vector used during training
#    4. RF model → Rise / Fall + confidence %
#    5. Return full result dict to Flask
# ═══════════════════════════════════════════════════════════════

import os
import re
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "rf_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# Cache the loaded model in memory so Flask doesn't reload it every request
_rf_model   = None
_rf_metrics = None


def get_model():
    """Load RF model once and cache it."""
    global _rf_model
    if _rf_model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        print("[predict] Loading RF model from disk...")
        _rf_model = joblib.load(MODEL_PATH)
        print("[predict] RF model loaded.")
    return _rf_model


def get_metrics():
    """Load saved training metrics."""
    global _rf_metrics
    if _rf_metrics is None:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                _rf_metrics = json.load(f)
    return _rf_metrics


# ════════════════════════════════════════════════════
#  FEATURE BUILDER
#  Must produce the EXACT same features as train.py
#  step 4 — same names, same order
# ════════════════════════════════════════════════════
def build_feature_vector(
    tweet_text:        str,
    human_sentiment:   float = 0.0,   # -1 / 0 / +1  (if known)
    has_human_label:   float = 0.0,   # 1 if human label provided
    known_pumper:      float = 0.0,
    price_region:      float = 0.0,
    inflection_point:  float = 0.0,
    tweet_volume:      float = 0.0,
    hour:              int   = 12,    # hour of day 0-23
    dayofweek:         int   = 0,     # 0=Mon … 6=Sun
    stock:             str   = "UNKNOWN",
    history_polarities: list = None,  # recent polarity history for rolling avg
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame matching training features exactly.
    Returns a DataFrame with one row — ready for rf.predict().
    """

    is_weekend      = 1.0 if dayofweek >= 5 else 0.0
    is_market_hours = 1.0 if (9 <= hour <= 16 and not is_weekend) else 0.0

    # Rolling averages — use history if provided, else repeat current
    if history_polarities and len(history_polarities) > 0:
        recent = history_polarities[-20:]  # last 20 values
    else:
        recent = [human_sentiment]

    def rolling_mean(n):
        vals = recent[-n:] if len(recent) >= n else recent
        return float(np.mean(vals)) if vals else 0.0

    human_diff = (
        human_sentiment - recent[-2]
        if len(recent) >= 2 else 0.0
    )

    text = str(tweet_text)

    features = {
        # ── Human label features ──
        "human_sentiment":   float(human_sentiment),
        "has_human_label":   float(has_human_label),
        "known_pumper":      float(known_pumper),
        "price_region":      float(price_region),
        "inflection_point":  float(inflection_point),
        "tweet_volume":      float(tweet_volume),

        # ── Rolling averages ──
        "human_roll_3":      rolling_mean(3),
        "human_roll_7":      rolling_mean(7),
        "human_roll_20":     rolling_mean(20),
        "human_diff":        float(human_diff),

        # ── Text statistics ──
        "tweet_length":      float(len(text)),
        "word_count":        float(len(text.split())),
        "exclaim_count":     float(text.count("!")),
        "question_count":    float(text.count("?")),
        "caps_ratio":        float(
                                 len(re.findall(r"[A-Z]", text)) /
                                 max(len(text), 1)
                             ),
        "has_cashtag":       float(bool(re.search(r"\$[A-Z]+", text))),
        "has_url":           float("http" in text.lower()),
        "has_number":        float(bool(re.search(r"\d+", text))),

        # ── Time features ──
        "hour":              float(hour),
        "dayofweek":         float(dayofweek),
        "is_weekend":        is_weekend,
        "is_market_hours":   is_market_hours,
    }

    # ── Stock dummies ──
    # These must match the top-10 stocks from training
    # We load feature names from the saved metrics to know which stocks exist
    metrics = get_metrics()
    if metrics and "feature_names" in metrics:
        for fname in metrics["feature_names"]:
            if fname.startswith("stk_"):
                # e.g. stk_AAPL → check if stock matches
                stk_name = fname[4:]  # remove "stk_"
                clean_stock = re.sub(r"[^A-Za-z0-9]", "_", stock).upper()[:12]
                features[fname] = 1.0 if clean_stock == stk_name else 0.0

    # ── Technical proxies (from human_sentiment, same as training) ──
    hs_norm = float(np.clip(human_sentiment, -1, 1))
    features["rsi_proxy"]       = 50 + hs_norm * 25
    features["macd_proxy"]      = hs_norm * 0.4
    features["sma_ratio_proxy"] = 1.0 + hs_norm * 0.03
    features["volume_proxy"]    = abs(hs_norm) * 0.08

    # ── Build DataFrame with correct column order from training ──
    if metrics and "feature_names" in metrics:
        # Use exact same column order as training
        ordered = {}
        for fname in metrics["feature_names"]:
            ordered[fname] = features.get(fname, 0.0)
        return pd.DataFrame([ordered])
    else:
        return pd.DataFrame([features])


# ════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION
#  Called by Flask routes
# ════════════════════════════════════════════════════
def predict_stock(
    tweet_text:       str,
    stock:            str   = "UNKNOWN",
    human_sentiment:  float = None,    # pass if known, else FinBERT generates it
    known_pumper:     float = 0.0,
    price_region:     float = 0.0,
    inflection_point: float = 0.0,
    tweet_volume:     float = 0.0,
    hour:             int   = None,
    dayofweek:        int   = None,
    history:          list  = None,
) -> dict:
    """
    Full prediction pipeline:
      1. FinBERT sentiment on tweet_text
      2. Build feature vector
      3. RF prediction

    Returns complete result dict.
    """
    import datetime
    from model import classify_text

    # ── Step 1: FinBERT sentiment ──
    finbert_result = classify_text(tweet_text)
    fb_label    = finbert_result["label"]       # "positive"/"neutral"/"negative"
    fb_score    = finbert_result["score"]
    fb_score_pct= finbert_result["score_pct"]
    fb_polarity = finbert_result["polarity"]    # +0.94 / 0.0 / -0.94

    # ── Step 2: Resolve human_sentiment ──
    # If caller provides human_sentiment, use it.
    # Otherwise derive from FinBERT label as a proxy.
    if human_sentiment is None:
        if fb_label == "positive":
            human_sentiment = 1.0
        elif fb_label == "negative":
            human_sentiment = -1.0
        else:
            human_sentiment = 0.0
        has_human_label = 0.0   # derived, not truly human
    else:
        has_human_label = 1.0   # caller provided real label

    # ── Step 3: Time defaults ──
    now = datetime.datetime.now()
    if hour     is None: hour     = now.hour
    if dayofweek is None: dayofweek = now.weekday()

    # ── Step 4: Build feature vector ──
    X = build_feature_vector(
        tweet_text       = tweet_text,
        human_sentiment  = human_sentiment,
        has_human_label  = has_human_label,
        known_pumper     = known_pumper,
        price_region     = price_region,
        inflection_point = inflection_point,
        tweet_volume     = tweet_volume,
        hour             = hour,
        dayofweek        = dayofweek,
        stock            = stock,
        history_polarities = history or [],
    )

    # ── Step 5: RF prediction ──
    rf = get_model()

    if rf is None:
        # Model not trained yet — fallback rule-based
        score = fb_polarity * 0.6 + human_sentiment * 0.4
        is_rise    = score >= 0
        confidence = min(85, max(52, int(55 + abs(score) * 30)))
        return {
            "prediction":    "Rise" if is_rise else "Fall",
            "confidence":    confidence,
            "prob_rise":     confidence / 100 if is_rise else (100 - confidence) / 100,
            "prob_fall":     (100 - confidence) / 100 if is_rise else confidence / 100,
            "model_ready":   False,
            "finbert":       _finbert_block(fb_label, fb_score, fb_score_pct, fb_polarity),
            "features_used": list(X.columns),
            "note":          "Rule-based fallback — rf_model.joblib not found",
        }

    pred       = rf.predict(X)[0]           # 0 or 1
    proba      = rf.predict_proba(X)[0]     # [prob_fall, prob_rise]
    prob_rise  = round(float(proba[1]), 4)
    prob_fall  = round(float(proba[0]), 4)
    confidence = int(max(prob_rise, prob_fall) * 100)
    label      = "Rise" if pred == 1 else "Fall"

    # ── Step 6: Build signal interpretation ──
    signal = _interpret_signal(label, confidence, fb_label, fb_polarity)

    return {
        # Core prediction
        "prediction":    label,
        "confidence":    confidence,
        "prob_rise":     prob_rise,
        "prob_fall":     prob_fall,
        "model_ready":   True,

        # FinBERT result
        "finbert": _finbert_block(fb_label, fb_score, fb_score_pct, fb_polarity),

        # Signal interpretation for dashboard
        "signal":        signal["text"],
        "signal_color":  signal["color"],
        "action":        signal["action"],

        # Stock & context
        "stock":         stock,
        "features_used": list(X.columns),
        "n_features":    len(X.columns),
    }


def _finbert_block(label, score, score_pct, polarity):
    return {
        "label":     label,
        "score":     score,
        "score_pct": score_pct,
        "polarity":  polarity,
        "emoji":     "🟢" if label == "positive" else
                     ("🔴" if label == "negative" else "🟡"),
    }


def _interpret_signal(prediction, confidence, fb_label, fb_polarity):
    """Human-readable signal interpretation for the dashboard."""
    if prediction == "Rise" and confidence >= 75:
        return {
            "text":   "Strong buy signal",
            "color":  "green",
            "action": "Sentiment strongly positive — corroborates upward movement"
        }
    elif prediction == "Rise" and confidence >= 60:
        return {
            "text":   "Moderate buy signal",
            "color":  "green",
            "action": "Positive sentiment detected — consider as supporting evidence"
        }
    elif prediction == "Fall" and confidence >= 75:
        return {
            "text":   "Strong sell signal",
            "color":  "red",
            "action": "Sentiment strongly negative — may indicate downward pressure"
        }
    elif prediction == "Fall" and confidence >= 60:
        return {
            "text":   "Moderate sell signal",
            "color":  "red",
            "action": "Negative sentiment detected — monitor closely"
        }
    else:
        return {
            "text":   "Uncertain signal",
            "color":  "amber",
            "action": "Conflicting signals — do not act on prediction alone"
        }


# ════════════════════════════════════════════════════
#  BATCH PREDICTION
#  Predict for multiple tweets at once
# ════════════════════════════════════════════════════
def predict_batch(tweets: list, stock: str = "UNKNOWN") -> list:
    """
    Predict for a list of tweet texts.
    Returns list of prediction dicts.
    """
    results    = []
    history    = []   # rolling history of human_sentiment values

    for tweet in tweets:
        result = predict_stock(
            tweet_text = tweet,
            stock      = stock,
            history    = history.copy(),
        )
        # update rolling history
        history.append(result["finbert"]["polarity"])
        if len(history) > 20:
            history = history[-20:]
        results.append(result)

    return results


# ════════════════════════════════════════════════════
#  MODEL INFO  — returns training metrics for dashboard
# ════════════════════════════════════════════════════
def get_model_info() -> dict:
    """Return training metrics for the dashboard KPI cards."""
    metrics = get_metrics()
    rf      = get_model()

    return {
        "model_ready":    rf is not None,
        "cv_accuracy":    metrics.get("cv_accuracy",   0) if metrics else 0,
        "cv_std":         metrics.get("cv_std",        0) if metrics else 0,
        "test_accuracy":  metrics.get("test_accuracy", 0) if metrics else 0,
        "auc_roc":        metrics.get("auc_roc",       0) if metrics else 0,
        "f1_weighted":    metrics.get("f1_weighted",   0) if metrics else 0,
        "precision":      metrics.get("precision",     0) if metrics else 0,
        "recall":         metrics.get("recall",        0) if metrics else 0,
        "tp":             metrics.get("tp", 0) if metrics else 0,
        "tn":             metrics.get("tn", 0) if metrics else 0,
        "fp":             metrics.get("fp", 0) if metrics else 0,
        "fn":             metrics.get("fn", 0) if metrics else 0,
        "total_samples":  metrics.get("total_samples", 0) if metrics else 0,
        "n_features":     metrics.get("n_features",   0) if metrics else 0,
        "feature_importances": metrics.get("feature_importances", {}) if metrics else {},
        "feature_names":  metrics.get("feature_names", []) if metrics else [],
        "n_estimators":   rf.n_estimators if rf else 0,
        "max_depth":      rf.max_depth    if rf else 0,
    }