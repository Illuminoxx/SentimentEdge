# SentimentEdge 📈

Financial sentiment analysis combining **FinBERT + Random Forest**
for next-day stock movement prediction.

## 🔴 Live Demo
👉 vectorxx-sentiment.hf.space

## Stack
| Layer | Tech |
|---|---|
| Sentiment model | ProsusAI/FinBERT |
| Movement classifier | Random Forest (scikit-learn) |
| Data source | historical tweets data |
| Backend | Flask |
| Deployment | Hugging Face Spaces (Docker) |

## How it works
1. demand financial text at real time as input by user
2. Runs each word through FinBERT for sentiment scoring
3. Combines sentiment features with price/volume data
4. Random Forest predicts next-day movement (Rise/Fall)

## Run Locally (fully offline)
-git clone https://github.com/Illuminoxx/SentimentEdge.git
-cd SentimentEdge
-pip install -r requirements.txt
-python backend/app.py
