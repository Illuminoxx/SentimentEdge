# ═══════════════════════════════════════════════════════════════
#  app.py  —  SentimentEdge Flask Backend
#
#  Start:  python app.py
#  URL:    http://localhost:5000
#
#  Routes:
#    GET  /api/status          → server health + model info
#    GET  /api/metrics         → full training metrics for dashboard
#    POST /api/analyze         → FinBERT only (text → sentiment)
#    POST /api/predict         → FinBERT + RF (text → Rise/Fall)
#    POST /api/predict/batch   → multiple tweets at once
# ═══════════════════════════════════════════════════════════════

import os
import time
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)   # allow dashboard HTML to call this API

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")

@app.route("/finbert")
def finbert():
    return render_template("finbert.html")

@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")

# Import our modules
from model   import classify_text, classify_batch
from predict import predict_stock, predict_batch, get_model_info


# ════════════════════════════════════════════════════
#  GET /api/status
#  Health check — dashboard calls this on load
# ════════════════════════════════════════════════════
@app.route("/api/status", methods=["GET"])
def status():
    info = get_model_info()
    return jsonify({
        "status":       "online",
        "finbert":      "ready",
        "rf_model":     "ready" if info["model_ready"] else "not_trained",
        "test_accuracy": info["test_accuracy"],
        "auc_roc":       info["auc_roc"],
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# ════════════════════════════════════════════════════
#  GET /api/metrics
#  Returns all training metrics for dashboard KPI cards
# ════════════════════════════════════════════════════
@app.route("/api/metrics", methods=["GET"])
def metrics():
    info = get_model_info()
    return jsonify(info)


# ════════════════════════════════════════════════════
#  POST /api/analyze
#  FinBERT only — classify sentiment of any text
#  Body: { "text": "AAPL earnings crushed it" }
# ════════════════════════════════════════════════════
@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        # Single text
        if "text" in data:
            text = str(data["text"]).strip()
            if not text:
                return jsonify({"error": "text cannot be empty"}), 400

            result = classify_text(text)
            return jsonify({
                "input":     text,
                "label":     result["label"],
                "score":     result["score"],
                "score_pct": result["score_pct"],
                "polarity":  result["polarity"],
                "emoji":     "🟢" if result["label"] == "positive"
                             else ("🔴" if result["label"] == "negative" else "🟡"),
            })

        # Batch of texts
        elif "texts" in data:
            texts = data["texts"]
            if not isinstance(texts, list) or len(texts) == 0:
                return jsonify({"error": "texts must be a non-empty list"}), 400

            results = classify_batch(texts)
            output  = []
            for text, r in zip(texts, results):
                output.append({
                    "input":     text,
                    "label":     r["label"],
                    "score":     r["score"],
                    "score_pct": r["score_pct"],
                    "polarity":  r["polarity"],
                })
            return jsonify({"results": output, "count": len(output)})

        else:
            return jsonify({"error": "Send 'text' or 'texts' in body"}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════
#  POST /api/predict
#  Full pipeline: FinBERT + RF → Rise/Fall prediction
#
#  Body (required):
#    { "text": "TSLA margins collapsing fast" }
#
#  Body (optional extras for better prediction):
#    {
#      "text":             "AAPL earnings beat",
#      "stock":            "AAPL",
#      "human_sentiment":  1,       // if you know: 1/0/-1
#      "known_pumper":     0,
#      "price_region":     1,       // 1=buy 0=hold -1=sell
#      "inflection_point": 0,
#      "tweet_volume":     0,
#      "hour":             14,      // 0-23
#      "dayofweek":        1,       // 0=Mon 6=Sun
#      "history":          [0.5, 0.3, 0.7]  // recent polarities
#    }
# ════════════════════════════════════════════════════
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        result = predict_stock(
            tweet_text       = text,
            stock            = str(data.get("stock", "UNKNOWN")),
            human_sentiment  = data.get("human_sentiment", None),
            known_pumper     = float(data.get("known_pumper",     0)),
            price_region     = float(data.get("price_region",     0)),
            inflection_point = float(data.get("inflection_point", 0)),
            tweet_volume     = float(data.get("tweet_volume",     0)),
            hour             = data.get("hour",      None),
            dayofweek        = data.get("dayofweek", None),
            history          = data.get("history",   []),
        )

        return jsonify({
            **result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════
#  POST /api/predict/batch
#  Predict for multiple tweets at once
#  Body: { "tweets": ["tweet1", "tweet2"], "stock": "AAPL" }
# ════════════════════════════════════════════════════
@app.route("/api/predict/batch", methods=["POST"])
def predict_batch_route():
    try:
        data   = request.get_json()
        tweets = data.get("tweets", [])
        stock  = str(data.get("stock", "UNKNOWN"))

        if not tweets or not isinstance(tweets, list):
            return jsonify({"error": "tweets must be a non-empty list"}), 400

        results    = predict_batch(tweets, stock=stock)
        rises      = sum(1 for r in results if r["prediction"] == "Rise")
        falls      = len(results) - rises
        avg_conf   = round(sum(r["confidence"] for r in results) / len(results), 1)
        avg_pol    = round(sum(r["finbert"]["polarity"] for r in results) / len(results), 4)

        return jsonify({
            "stock":        stock,
            "total":        len(results),
            "predictions":  results,
            "summary": {
                "rise_count":    rises,
                "fall_count":    falls,
                "avg_confidence": avg_conf,
                "avg_polarity":  avg_pol,
                "overall":       "Rise" if rises > falls else "Fall",
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════
#  START SERVER
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    print("\n" + "="*55)
    print("  SentimentEdge Backend")
    print("="*55)
    print(f"  Server  →  http://localhost:{port}")
    print(f"  Status  →  http://localhost:{port}/api/status")
    print(f"  Metrics →  http://localhost:{port}/api/metrics")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
