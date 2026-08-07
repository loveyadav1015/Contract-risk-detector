# Contract Risk Detector

An ML-powered tool that analyzes contract clauses and classifies them into risk levels (**Low**, **Medium**, **High**) using fine-tuned Transformer models (LegalBERT / DeBERTa) on the [CUAD dataset](https://www.atticusprojectai.org/cuad).

---

## Project Description

This project fine-tunes a pretrained legal language model on the Contract Understanding Atticus Dataset (CUAD) to automatically detect and classify risky clauses in legal contracts. It exposes a FastAPI service for real-time contract analysis and optionally supports a RAG (Retrieval-Augmented Generation) layer for clause-level Q&A.

### Key Features

- **Clause-level risk classification** — Predicts Low / Medium / High risk for individual contract clauses
- **Full contract analysis** — Accepts PDF uploads and analyzes all extracted clauses
- **REST API** — FastAPI-based service for integration into downstream applications
- **RAG module (optional)** — Semantic search over clause embeddings with FAISS + LLM-based Q&A

---

## Architecture

```
contract-risk-detector/
├── data/              # Raw and processed CUAD data
├── notebooks/         # Jupyter notebooks for EDA
├── src/               # Core ML code (dataset, model, train, evaluate, predict)
├── api/               # FastAPI application (routes, services)
├── rag/               # Optional RAG module (embedder, retriever, Q&A)
├── models/            # Saved model checkpoints
├── configs/           # YAML configuration
└── tests/             # Unit and integration tests
```

---

## Setup

### Prerequisites

- Python 3.11+
- (Optional) NVIDIA GPU with CUDA for training

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd contract-risk-detector

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Data Setup

<!-- TODO: Add instructions for downloading and placing CUAD dataset files -->

Place the CUAD dataset files in `data/raw/`. See `data/README.md` for details.

---

## How to Run

### Training

<!-- TODO: Add exact training command after train.py is implemented -->

```bash
python -m src.train --config configs/config.yaml
```

### Evaluation

<!-- TODO: Add exact evaluation command after evaluate.py is implemented -->

```bash
python -m src.evaluate --config configs/config.yaml
```

### API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t contract-risk-detector .
docker run -p 8000:8000 contract-risk-detector
```

---

## API Docs

Once the server is running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| GET    | `/health`          | Service health check                     |
| POST   | `/analyze`         | Upload a contract (PDF/text) for analysis|
| POST   | `/analyze/clause`  | Analyze a single clause string           |

<!-- TODO: Add request/response examples after API is implemented -->

---

## Testing

```bash
pytest tests/
```

---

## License

<!-- TODO: Add license information -->
