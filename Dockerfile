FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY index.html .
COPY style.css .
COPY app.js .
COPY pages/ ./pages/

EXPOSE 7860
ENV PORT=7860

CMD ["python", "backend/app.py"]