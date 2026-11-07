# Lightweight Python 3.10 slim base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies (build-essential for C extensions if needed, git for huggingface cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .

# Install CPU version of PyTorch first to keep the image lean (~150MB instead of ~2GB CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model to prevent cold-start latency in production
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Launch Streamlit with environment variable port binding (works on Render, Railway, Cloud Run, HF)
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true

