# Dockerfile
# Python app container for the GDELT RAG pipeline

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace models at build time so they're baked in
# (saves time every time you restart the container)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading BGE-M3 embedding model...'); \
SentenceTransformer('BAAI/bge-m3'); \
print('Downloading BGE reranker...'); \
CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512); \
print('All models downloaded!')"

# Copy project code
COPY . .

# Default command — can be overridden in docker-compose
CMD ["python", "chatbot/app.py"]