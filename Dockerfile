# API + frontend in one image. Train first so data/artifacts/ exists:
#   python -m src.train && docker build -t howlong . && docker run -p 8000:8000 howlong
FROM python:3.12-slim

# xgboost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY frontend/ frontend/
COPY data/artifacts/ data/artifacts/

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
