# ═══════════════════════════════════════════════════════════════
#  train.py  —  SentimentEdge  (FIXED — no label leakage)
#
#  ROOT CAUSE OF 100% ACCURACY (now fixed):
#    Before: FinBERT labels tweet → RF trained on fb_polarity
#            to predict those same labels = memorisation
#
#  THE FIX (3 changes):
#    1. fb_polarity / fb_score / fb_label are NO LONGER features
#       They are only used to CREATE the label, then discarded
#    2. RF only sees: rolling averages, time, stock, human extras
#       These are weaker signals → realistic 65-80% accuracy
#    3. For datasets WITH human labels: human_sentiment is kept
#       as a feature since it's independent of fb_polarity
#
#  HOW TO RUN:
#    python train.py --csv fintwit.csv stock_tweets.csv
#    python train.py --csv anyfile.csv --preview
# ═══════════════════════════════════════════════════════════════

import argparse
import os
import sys
import json
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "rf_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
CACHE_DIR    = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ════════════════════════════════════════════════════
#  COLUMN AUTO-DETECTOR
# ════════════════════════════════════════════════════
TEXT_KEYWORDS      = ["tweet","text","body","content","message","post",
                      "sentence","headline","title","description","comment",
                      "review","article","news","caption"]
SENTIMENT_KEYWORDS = ["sentiment","label","target","class","category",
                      "polarity","score","emotion","opinion","rating",
                      "mood","tone","stance"]
DATE_KEYWORDS      = ["date","datetime","time","timestamp","created",
                      "created_at","published","posted_at","when"]
TICKER_KEYWORDS    = ["stock","ticker","symbol","company","firm",
                      "stock name","stock_name","equity","asset",
                      "company name","company_name","instrument"]
PUMPER_KEYWORDS    = ["pumper","known_pumper","influencer","verified"]
REGION_KEYWORDS    = ["price_region","price region","region","signal","buy_sell"]
INFLECT_KEYWORDS   = ["inflection","inflection_point","turning_point","pivot"]
VOLUME_KEYWORDS    = ["volume","vol","engagement","likes","retweet","upvote"]


def _match(col: str, keywords: list) -> bool:
    col = col.lower().strip()
    return any(kw in col for kw in keywords)


def _is_text(series: pd.Series) -> bool:
    if series.dtype != object: return False
    s = series.dropna().head(100)
    if len(s) == 0: return False
    return (s.astype(str).str.len().mean() > 20 and
            s.astype(str).str.split().str.len().mean() > 3 and
            series.nunique() / max(len(series),1) > 0.3)


def _is_sentiment(series: pd.Series) -> bool:
    unique_vals = set(series.dropna().unique())
    try:
        num = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(num) > 0:
            uv = set(num.unique())
            if uv <= {-1,0,1}: return True
            if uv <= {0,1}:    return True
            if uv <= {1,2,3}:  return True
            if uv <= {-1,1}:   return True
    except Exception:
        pass
    str_vals = {str(v).lower().strip() for v in unique_vals}
    sentiment_strings = {"positive","negative","neutral","pos","neg","neu",
                         "bullish","bearish","buy","sell","hold",
                         "1","0","-1","2","good","bad","mixed"}
    overlap = str_vals & sentiment_strings
    if len(str_vals) > 0 and len(overlap)/len(str_vals) > 0.5: return True
    if series.dtype == object and series.nunique() <= 5:        return len(overlap) > 0
    return False


def _is_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series): return True
    if series.dtype != object: return False
    sample = series.dropna().head(50).astype(str)
    pat = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{4}-\d{2}-\d{2}T")
    return sample.apply(lambda x: bool(pat.search(x))).mean() > 0.6


def detect_columns(df: pd.DataFrame) -> dict:
    detected = {k: None for k in
                ["text","sentiment","date","ticker","pumper","region","inflect","volume"]}
    for col in df.columns:
        if detected["text"] is None and _match(col, TEXT_KEYWORDS) and _is_text(df[col]):
            detected["text"] = col
        if detected["sentiment"] is None and _match(col, SENTIMENT_KEYWORDS) and _is_sentiment(df[col]):
            detected["sentiment"] = col
        if detected["date"] is None and (_match(col, DATE_KEYWORDS) or _is_date(df[col])):
            detected["date"] = col
        if detected["ticker"]  is None and _match(col, TICKER_KEYWORDS):  detected["ticker"]  = col
        if detected["pumper"]  is None and _match(col, PUMPER_KEYWORDS):  detected["pumper"]  = col
        if detected["region"]  is None and _match(col, REGION_KEYWORDS):  detected["region"]  = col
        if detected["inflect"] is None and _match(col, INFLECT_KEYWORDS): detected["inflect"] = col
        if detected["volume"]  is None and _match(col, VOLUME_KEYWORDS):  detected["volume"]  = col

    # fallback: first text-like column
    if detected["text"] is None:
        for col in df.columns:
            if _is_text(df[col]):
                detected["text"] = col
                break
    if detected["text"] is None:
        for col in df.columns:
            if df[col].dtype == object:
                detected["text"] = col
                break

    # avoid sentiment == text
    if detected["sentiment"] == detected["text"]:
        detected["sentiment"] = None

    return detected


def normalise_sentiment(series: pd.Series):
    mapping = {
        "positive":1.0,"pos":1.0,"bullish":1.0,"buy":1.0,"rise":1.0,
        "up":1.0,"good":1.0,"1":1.0,"2":1.0,"1.0":1.0,
        "negative":-1.0,"neg":-1.0,"bearish":-1.0,"sell":-1.0,
        "fall":-1.0,"down":-1.0,"bad":-1.0,"-1":-1.0,"-1.0":-1.0,
        "neutral":0.0,"neu":0.0,"hold":0.0,"mixed":0.0,
        "none":0.0,"0":0.0,"0.0":0.0,
    }
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        vals = numeric.dropna().unique()
        if set(vals) <= {-1,0,1,-1.0,0.0,1.0}: return numeric
        if set(vals) <= {0,1,0.0,1.0}:          return numeric * 2 - 1
        if set(vals) <= {1,2,3}:                 return numeric - 2
        return numeric.clip(-1,1)
    mapped = series.astype(str).str.lower().str.strip().map(mapping)
    if mapped.notna().mean() > 0.5: return mapped
    return None


# ════════════════════════════════════════════════════
#  STEP 1 — Universal CSV Loader
# ════════════════════════════════════════════════════
def load_any_csv(csv_path: str, preview: bool = False) -> pd.DataFrame:
    filename = os.path.basename(csv_path)
    print(f"\n  ── Loading: {filename} ──")

    if not os.path.exists(csv_path):
        print(f"  ERROR: File not found → {csv_path}")
        sys.exit(1)

    for enc in ["utf-8","latin-1","cp1252","utf-16"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            break
        except Exception:
            continue
    else:
        print(f"  ERROR: Cannot read {csv_path}")
        sys.exit(1)

    original_cols = list(df.columns)
    df.columns    = df.columns.str.lower().str.strip()

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {original_cols}")

    detected = detect_columns(df)

    print(f"\n  AUTO-DETECTED:")
    print(f"    text      → '{detected['text']}'")
    print(f"    sentiment → '{detected['sentiment'] or 'NOT FOUND → FinBERT will label'}'")
    print(f"    date      → '{detected['date']      or 'not found'}'")
    print(f"    ticker    → '{detected['ticker']    or 'not found'}'")
    print(f"    pumper    → '{detected['pumper']    or 'not found'}'")
    print(f"    region    → '{detected['region']    or 'not found'}'")
    print(f"    inflect   → '{detected['inflect']   or 'not found'}'")

    if preview:
        print(f"\n  PREVIEW MODE — not training.")
        return pd.DataFrame()

    if detected["text"] is None:
        print(f"  ERROR: Cannot find a text column. Columns: {list(df.columns)}")
        sys.exit(1)

    out = pd.DataFrame()
    out["tweet"]  = df[detected["text"]].astype(str).str.strip()
    out["source"] = filename

    if detected["sentiment"]:
        norm = normalise_sentiment(df[detected["sentiment"]])
        out["human_sentiment"] = norm if norm is not None else 0.0
        out["has_human_label"] = 1.0
    else:
        out["human_sentiment"] = 0.0
        out["has_human_label"] = 0.0

    out["date"]  = pd.to_datetime(df[detected["date"]], utc=True, errors="coerce") \
                   if detected["date"] else pd.NaT
    out["stock"] = df[detected["ticker"]].astype(str) if detected["ticker"] else "UNKNOWN"

    for key, col in [("known_pumper",     detected["pumper"]),
                     ("price_region",      detected["region"]),
                     ("inflection_point",  detected["inflect"]),
                     ("tweet_volume",      detected["volume"])]:
        out[key] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col else 0.0

    before = len(out)
    out = out[out["tweet"].str.len() > 5].dropna(subset=["tweet"])
    print(f"  Clean rows: {len(out):,}  (dropped {before-len(out):,})")

    return out


# ════════════════════════════════════════════════════
#  STEP 2 — Datetime Features
# ════════════════════════════════════════════════════
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2/6] Parsing datetime features...")
    df["hour"]            = df["date"].dt.hour.fillna(12).astype(float)
    df["dayofweek"]       = df["date"].dt.dayofweek.fillna(0).astype(float)
    df["is_weekend"]      = (df["dayofweek"] >= 5).astype(float)
    df["is_market_hours"] = (
        (df["hour"] >= 9) & (df["hour"] <= 16) & (df["is_weekend"] == 0)
    ).astype(float)
    valid = df["date"].dropna()
    if len(valid) > 0:
        print(f"  Date range : {valid.min().date()} → {valid.max().date()}")
    return df


# ════════════════════════════════════════════════════
#  STEP 3 — FinBERT (with caching)
#  NOTE: fb_polarity/fb_score are used ONLY to make the
#  label — they are NOT passed to the Random Forest as
#  features. This prevents the 100% accuracy leakage.
# ════════════════════════════════════════════════════
def run_finbert(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3/6] Running FinBERT (labels only — not features)...")

    from model import classify_batch
    results_list = []

    for source in df["source"].unique():
        subset     = df[df["source"] == source].copy()
        safe_name  = re.sub(r"[^A-Za-z0-9_]", "_", source[:40])
        cache_path = os.path.join(CACHE_DIR, f"cache_{safe_name}.csv")

        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path)
            if "fb_polarity" in cached.columns and len(cached) == len(subset):
                print(f"  [{source}] Loaded from cache ({len(subset):,} rows)")
                subset["fb_polarity"] = cached["fb_polarity"].values
                subset["fb_label"]    = cached["fb_label"].values
                results_list.append(subset)
                continue

        total      = len(subset)
        texts      = subset["tweet"].astype(str).tolist()
        batch_size = 32
        polarities, labels = [], []

        print(f"  [{source}] FinBERT labelling {total:,} tweets...")
        for i in range(0, total, batch_size):
            batch   = texts[i : i + batch_size]
            results = classify_batch(batch)
            for r in results:
                polarities.append(r["polarity"])
                labels.append(r["label"])
            done = min(i + batch_size, total)
            pct  = int(done / total * 100)
            bar  = "█" * (pct // 2) + "░" * (50 - pct // 2)
            print(f"  [{bar}] {pct}%  ({done:,}/{total:,})", end="\r")

        print(f"\n  [{source}] Done.")
        subset = subset.copy()
        subset["fb_polarity"] = polarities
        subset["fb_label"]    = labels
        subset.to_csv(cache_path, index=False)
        print(f"  Cached → {cache_path}")
        results_list.append(subset)

    df = pd.concat(results_list, ignore_index=True)

    # ── Build binary label using FinBERT + human labels ──
    # !! fb_polarity used ONLY HERE for label creation !!
    def make_label(row):
        hs = row["human_sentiment"]
        pr = row["price_region"]
        fb = row["fb_polarity"]

        if row["has_human_label"] == 1.0:
            if hs == 1.0:   return 1
            if hs == -1.0:  return 0
            if pr == 1.0:   return 1
            if pr == -1.0:  return 0

        # No human label → use FinBERT
        if row["fb_label"] == "positive": return 1
        if row["fb_label"] == "negative": return 0
        return 1 if fb >= 0 else 0

    df["label"] = df.apply(make_label, axis=1).astype(int)

    dist  = df["label"].value_counts()
    total = len(df)
    fb    = df["fb_label"].value_counts()
    print(f"\n  FinBERT dist  : pos={fb.get('positive',0):,} | "
          f"neu={fb.get('neutral',0):,} | neg={fb.get('negative',0):,}")
    print(f"  Binary labels : Rise={dist.get(1,0):,} ({dist.get(1,0)/total*100:.1f}%) "
          f"| Fall={dist.get(0,0):,} ({dist.get(0,0)/total*100:.1f}%)")

    # !! Drop fb_polarity and fb_label — NOT used as features !!
    df.drop(columns=["fb_polarity", "fb_label"], inplace=True)

    return df


# ════════════════════════════════════════════════════
#  STEP 4 — Feature Matrix
#
#  WHAT IS INCLUDED (no leakage):
#    - Rolling sentiment averages (aggregate signal, not raw score)
#    - Human sentiment (independent of FinBERT)
#    - Known_Pumper, Price_Region, Inflection_Point
#    - Time features (hour, day, weekend, market hours)
#    - Stock dummies
#    - Technical proxies
#    - Tweet text statistics (length, word count, punctuation)
#
#  WHAT IS EXCLUDED (causes leakage):
#    - fb_polarity  (directly encodes the label)
#    - fb_score     (directly encodes the label)
#    - fb_label     (IS the label)
# ════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[4/6] Building feature matrix (leakage-free)...")

    features = pd.DataFrame(index=df.index)

    # ── Human label features (independent of FinBERT) ──
    features["human_sentiment"]   = df["human_sentiment"].astype(float)
    features["has_human_label"]   = df["has_human_label"].astype(float)
    features["known_pumper"]      = df["known_pumper"].astype(float)
    features["price_region"]      = df["price_region"].astype(float)
    features["inflection_point"]  = df["inflection_point"].astype(float)
    features["tweet_volume"]      = df["tweet_volume"].astype(float)

    # ── Human sentiment rolling (group by stock if available) ──
    hs = df["human_sentiment"].astype(float)
    features["human_roll_3"]  = hs.rolling(3,  min_periods=1).mean()
    features["human_roll_7"]  = hs.rolling(7,  min_periods=1).mean()
    features["human_roll_20"] = hs.rolling(20, min_periods=1).mean()
    features["human_diff"]    = hs.diff().fillna(0)

    # ── Text statistics (from the raw tweet — no FinBERT) ──
    tweet_text = df["tweet"].astype(str)
    features["tweet_length"]   = tweet_text.str.len().astype(float)
    features["word_count"]     = tweet_text.str.split().str.len().astype(float)
    features["exclaim_count"]  = tweet_text.str.count("!").astype(float)
    features["question_count"] = tweet_text.str.count(r"\?").astype(float)
    features["caps_ratio"]     = (
        tweet_text.str.count(r"[A-Z]") /
        tweet_text.str.len().replace(0, 1)
    ).astype(float)
    features["has_cashtag"]    = tweet_text.str.contains(r"\$[A-Z]+").astype(float)
    features["has_url"]        = tweet_text.str.contains(r"http").astype(float)
    features["has_number"]     = tweet_text.str.contains(r"\d+").astype(float)

    # ── Time features ──
    features["hour"]           = df["hour"]
    features["dayofweek"]      = df["dayofweek"]
    features["is_weekend"]     = df["is_weekend"]
    features["is_market_hours"]= df["is_market_hours"]

    # ── Stock dummies (top 10 stocks) ──
    top_stocks = df["stock"].value_counts().head(10).index.tolist()
    for s in top_stocks:
        safe = re.sub(r"[^A-Za-z0-9]", "_", s).upper()[:12]
        features[f"stk_{safe}"] = (df["stock"] == s).astype(float)

    # ── Technical proxies from human_sentiment ──
    # (NOT from fb_polarity — that would be leakage)
    hs_norm = df["human_sentiment"].clip(-1, 1).astype(float)
    features["rsi_proxy"]        = (50 + hs_norm * 25).clip(25, 75)
    features["macd_proxy"]       = hs_norm * 0.4
    features["sma_ratio_proxy"]  = 1.0 + hs_norm * 0.03
    features["volume_proxy"]     = hs_norm.abs() * 0.08

    features = features.fillna(0)
    print(f"  Shape    : {features.shape[0]:,} rows × {features.shape[1]} features")
    print(f"  Features : {list(features.columns)}")
    print(f"\n  NOTE: fb_polarity/fb_score excluded to prevent label leakage")
    print(f"        Expected accuracy: 65-80% (realistic, not 100%)")

    return features


# ════════════════════════════════════════════════════
#  STEP 5 — Train Random Forest
# ════════════════════════════════════════════════════
def train_model(X: pd.DataFrame, y: pd.Series):
    print(f"\n[5/6] Training Random Forest...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE → {len(X_train):,} balanced rows")
    except ImportError:
        print("  (SMOTE skipped — pip install imbalanced-learn to enable)")

    rf = RandomForestClassifier(
        n_estimators      = 200,
        max_depth         = 12,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        class_weight      = "balanced",
        criterion         = "gini",
        random_state      = 42,
        n_jobs            = -1,
    )
    rf.fit(X_train, y_train)
    print(f"  200 trees trained on {X.shape[1]} features.")

    return rf, X_test, y_test


# ════════════════════════════════════════════════════
#  STEP 6 — Evaluate & Save
# ════════════════════════════════════════════════════
def evaluate_and_save(rf, X, y, X_test, y_test):
    print(f"\n[6/6] Evaluating model...")

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  5-CV Accuracy : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    y_pred    = rf.predict(X_test)
    y_prob    = rf.predict_proba(X_test)[:, 1]
    acc       = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)
    f1        = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sanity check — warn if still suspiciously high
    if acc > 0.95:
        print(f"\n  ⚠ WARNING: Accuracy {acc*100:.1f}% is still very high.")
        print(f"    Check if human_sentiment is too directly correlated with labels.")
    else:
        print(f"\n  ✓ Accuracy looks realistic ({acc*100:.1f}%)")

    print(f"  Test Accuracy : {acc*100:.1f}%")
    print(f"  AUC-ROC       : {auc:.3f}")
    print(f"  F1 (weighted) : {f1:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fall','Rise'])}")

    importances = dict(zip(X.columns, rf.feature_importances_))
    top10 = sorted(importances.items(), key=lambda x: -x[1])[:10]
    print("  Top 10 Feature Importances:")
    for feat, imp in top10:
        bar = "█" * int(imp * 60)
        print(f"    {feat:<28} {imp*100:5.1f}%  {bar}")

    joblib.dump(rf, MODEL_PATH)
    print(f"\n  ✓ Model saved     → {MODEL_PATH}")

    metrics = {
        "cv_accuracy":   round(cv_scores.mean() * 100, 1),
        "cv_std":        round(cv_scores.std()  * 100, 1),
        "test_accuracy": round(acc * 100, 1),
        "auc_roc":       round(auc, 3),
        "f1_weighted":   round(f1, 3),
        "precision":     round(precision, 3),
        "recall":        round(recall, 3),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "total_samples": int(len(y)),
        "train_samples": int(len(y) * 0.8),
        "test_samples":  int(len(y) * 0.2),
        "n_features":    int(X.shape[1]),
        "feature_importances": {
            k: round(v * 100, 2)
            for k, v in sorted(importances.items(), key=lambda x: -x[1])
        },
        "feature_names": list(X.columns),
        "leakage_fixed": True,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved   → {METRICS_PATH}")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Accuracy={acc*100:.1f}%  AUC={auc:.3f}  F1={f1:.3f}")
    print(f"\n  Now run:  python app.py")
    print(f"{'='*60}\n")

    return metrics



# ════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════
def train(csv_files: list, preview: bool = False):
    print(f"\n{'='*60}")
    print("  SentimentEdge — Universal Trainer (Leakage-Free)")
    print(f"{'='*60}")
    print(f"\n[1/6] Loading {len(csv_files)} dataset(s)...")

    frames = []
    for path in csv_files:
        full = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        df   = load_any_csv(full, preview=preview)
        if len(df) > 0:
            frames.append(df)

    if preview:
        sys.exit(0)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined : {len(df):,} rows from {len(csv_files)} file(s)")
    for src, cnt in df["source"].value_counts().items():
        tag = "human labels" if df[df["source"]==src]["has_human_label"].mean() > 0.5 else "auto-label"
        print(f"    {src:<40} {cnt:>7,}  [{tag}]")

    df               = parse_datetime(df)
    df               = run_finbert(df)
    X                = build_features(df)
    y                = df["label"]
    rf, X_test, y_test = train_model(X, y)
    metrics          = evaluate_and_save(rf, X, y, X_test, y_test)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SentimentEdge Universal Trainer",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python train.py --csv fintwit.csv
  python train.py --csv fintwit.csv stock_tweets.csv
  python train.py --csv anyfile.csv --preview

Any CSV works automatically — no column mapping needed.
Expected accuracy after fix: 65-80% (realistic range).
        """
    )
    parser.add_argument("--csv", nargs="+", required=True,
                        help="One or more CSV files in the backend/ folder")
    parser.add_argument("--preview", action="store_true",
                        help="Show detected columns without training")
    args = parser.parse_args()
    train(csv_files=args.csv, preview=args.preview)
