FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY src/ src/
COPY rag/ rag/
COPY configs/ configs/
COPY models/ models/
COPY mini_transformer/ mini_transformer/

# Explicitly set HF_HOME to a writable directory for HF Spaces (non-root container)
ENV HF_HOME=/tmp/hf_cache

# Expose the HF Spaces required port (7860 instead of Render's dynamic 8000)
EXPOSE 7860

# Run the FastAPI server on port 7860 (HF Spaces specific)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 7860"]
