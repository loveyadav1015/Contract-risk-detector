# Contract Risk Detector 

## Mini-Transformer (From-Scratch) — Ablation Study

### Status
- Phase 1 (Tokenization & Vocabulary): ✅ Complete
- Phase 2 (Embeddings & Positional Encoding): ✅ Complete
- Phase 3 (Multi-Head Self-Attention): ✅ Complete
- Phase 4 (Encoder Block — Full Stack): ✅ Complete

### Purpose
Built as a separate architectural comparison to the fine-tuned LegalBERT pipeline above — implements Transformer internals (embeddings, multi-head self-attention, feed-forward, LayerNorm, residual connections) manually in raw PyTorch, without pretrained weights, to demonstrate understanding beyond fine-tuning. Trained on the same CUAD clause dataset for a fair, documented comparison.

### Phase 1 — Tokenization & Vocabulary
- Real vocab size: 5577
- Top tokens: the (27579), of (18184), to (15995), and (12741), or (11548), in (9111), any (6411), a (5637), agreement (5305), for (5162)
- Special tokens: PAD=0, UNK=1, CLS=2, SEP=3
- Sample encode/decode: encode `[2, 57, 105, 2346, 18, 29, 5, 4, 65, 5, 15, 12, 4, 12, 14, 19, 3515, 34, 40, 360, 362, 13, 90, 44, 84, 65, 13, 183, 6, 412, 569, 172, 237, 34, 4, 308, 65, 7, 122, 22, 82, 86, 153, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` → decode `<CLS> if distributor complies with all of the terms of this agreement the agreement shall be renewable on an annual basis for one 1 year terms for up to another ten 10 years on the same terms and conditions as set forth herein <SEP>`
- Verify: `python -m mini_transformer.tokenizer`

### Phase 2 — Embeddings & Positional Encoding
- Status: ✅ Complete
- d_model: 256
- Positional encoding: sinusoidal (fixed, non-learned)
- Real verified output shape: `(4, 64, 256)`
- Sample values `output[0, 0, :5]`: `[1.509156584739685, 8.360047340393066, 10.648882865905762, 13.468913078308105, -26.265277862548828]`
- NaN check: `torch.isnan(output).any(): False`
- PAD zero-check: `TokenEmbedding on all-PAD ids all zero: True` (`padding_idx=0` matches Phase 1 PAD id)
- Verify: `python -m mini_transformer.embeddings`

### Phase 3 — Multi-Head Self-Attention
- Status: ✅ Complete
- num_heads: 4, d_k per head: 64
- Real verified output shape: `(4, 64, 256)`
- Real verified attn_weights shape: `(4, 4, 64, 64)`
- NaN check: `torch.isnan(output).any(): False`
- Attention weights sum-to-1 check: `attn_weights[0, 0, 0, :].sum(): 1.0` (within 1e-4)
- Padding mask verification: first PAD key index 44 in sample 0; `attn_weights[0, 0, 0, 44]: 0.0`, `attn_weights[0, 0, 10, 44]: 0.0` (masked keys receive zero weight)
- Verify: `python -m mini_transformer.attention`

### Phase 4 — Encoder Block (Full Stack)
- Status: ✅ Complete
- num_layers: 4, d_ff: 1024
- Normalization: Post-LayerNorm (residual → LayerNorm)
- Real verified output shape: `(4, 64, 256)`
- Real total trainable parameters: 4,586,752
- NaN check: `torch.isnan(output).any(): False`
- Verify: `python -m mini_transformer.encoder`

### Phase 5 — Classification Head & Manual Training Loop
- Status: ✅ Complete
- Classifier: Linear(d_model=256, num_labels=3) on [CLS] token representation from final encoder layer
- Training: manual PyTorch loop, AdamW (lr=1e-4), CrossEntropyLoss, 10 epochs, batch_size=16
- Same train/val/test split as LegalBERT pipeline (seed=42, same 5800/725/726 split) — verified via matching split sizes
- Best checkpoint selected by highest val_f1_macro (epoch 9, ~0.81), saved to mini_transformer/best_model/mini_transformer_best.pt
- Total training+evaluation time: 1.58 minutes

### Phase 5 — Real Test Set Results
- Test Accuracy: 0.8168
- Test F1 Macro: 0.7984
- Test Precision Macro: 0.8016
- Test Recall Macro: 0.7971
- Per-class F1 — low: 0.82, medium: 0.88, high: 0.70

### Phase H — Regularization Experiment (Class Weights + Label Smoothing + Grad Clipping)
- Status: ✅ Complete
- Changes: class-weighted CrossEntropyLoss (weights: [0.8301, 0.8740, 1.5356]), label_smoothing=0.1, gradient clipping max_norm=1.0
- All other hyperparameters unchanged from baseline (Phase 5)

| Metric | Baseline (Phase 5) | With regularization (Phase H) |
|---|---|---|
| Test Accuracy | 0.8168 | 0.8306 |
| Test F1 Macro | 0.7984 | 0.8160 |
| High-risk F1 | 0.70 | 0.73 |

This regularization run improved overall test metrics, with accuracy increasing from 0.8168 to 0.8306 and macro F1 from 0.7984 to 0.8160. Most importantly for the target weakness, high-risk F1 improved from 0.70 to 0.73. The gain is modest but real, so this configuration is a better checkpoint than the original baseline.

### Final Comparison: Fine-tuned LegalBERT vs From-Scratch Mini-Transformer

| Metric | LegalBERT (fine-tuned) | Mini-Transformer (from-scratch) |
|---|---|---|
| Parameters | ~110,000,000 | 4,586,752 (~24x smaller) |
| Pretrained | Yes (legal-domain corpus) | No (random init) |
| Test Accuracy | 0.88 | 0.82 |
| Test F1 Macro | 0.87 | 0.80 |
| Low-risk F1 | 0.89 | 0.82 |
| Medium-risk F1 | 0.91 | 0.88 |
| High-risk F1 | 0.80 | 0.70 |

### Honest Takeaway
The from-scratch mini-transformer is ~24x smaller (4,586,752 vs ~110M parameters) and had no pretrained language knowledge, yet still reached 0.8168 accuracy and 0.7984 macro F1, which is within roughly 6-7 points of LegalBERT. That gap-to-baseline behavior is strong evidence the manual Transformer implementation is architecturally correct, because a broken attention/encoder stack would trend much closer to near-random performance for a 3-class task. The largest drop is on the high-risk class (0.80 → 0.70), which also has the fewest test examples (158), and this is where pretraining appears to help most because LegalBERT starts with language structure already learned while the from-scratch model must learn both language patterns and label mapping from limited labeled data. This mini-transformer should be treated as an architectural learning demonstration, not a production deployment candidate; the fine-tuned LegalBERT model remains the recommended serving model in the existing FastAPI backend.

### Known Limitations (Mini-Transformer)
- No separate history.json persisted (only training_curves.png plot and console logs record per-epoch metrics)
- Word-level tokenizer with UNK fallback (vs LegalBERT's WordPiece subword tokenizer) — likely loses information on rare/legal-specific terms not in the 5,577-token vocabulary
- Single training run — no hyperparameter search was performed (lr, d_model, num_layers were chosen as reasonable defaults, not tuned)
- Only 4 encoder layers vs LegalBERT's 12 — capacity-limited by design given the small dataset










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
├── api/                   # FastAPI application
│   ├── routes/            # API endpoints (analyze, analyze_mini, health)
│   └── services/          # Core logic (PDF text extraction, prediction services)
├── configs/               # Hyperparameter & YAML configurations
│   └── config.yaml        
├── data/                  # Dataset directory (.gitkeep tracked)
│   ├── raw/               # Downloaded CUAD dataset cache
│   └── processed/         # Processed clause records and PyTorch tensors
├── mini_transformer/      # Custom from-scratch Transformer module & ablation study
│   ├── attention.py       # Multi-Head Self-Attention implementation
│   ├── dataset.py         # Custom dataset & tokenization wrapper
│   ├── embeddings.py      # Token + Sinusoidal Positional Embeddings
│   ├── encoder.py         # Transformer Encoder block stack
│   ├── model.py           # Full MiniTransformerClassifier architecture
│   ├── tokenizer.py       # Custom BPE tokenizer loader/builder
│   ├── train_mini.py      # Manual PyTorch training & validation loop
│   └── best_model/        # Checkpoints, vocabulary, and performance plots
├── models/                # Saved fine-tuned model checkpoints (e.g., LegalBERT)
├── notebooks/             # Jupyter notebooks for data exploration and EDA
├── rag/                   # Contextual RAG module (embedder, retriever, Q&A)
├── src/                   # Core LegalBERT pipeline
│   ├── dataset.py         # CUAD parsing, tokenization, & DataLoaders
│   ├── evaluate.py        # Model evaluation & metrics calculation
│   ├── model.py           # LegalBERT sequence classification head setup
│   ├── predict.py         # Standalone inference logic
│   ├── train.py           # Training loop via HuggingFace Trainer
│   ├── utils.py           # Text cleaning, chunking, & logging utilities
│   └── visualize_training.py # Plotting training curves & confusion matrices
├── tests/                 # Unit and integration test suite
│   ├── test_api.py
│   ├── test_dataset.py
│   └── test_model.py
├── .env.example           # Environment variables template
├── .gitignore             # Ignores large datasets (*.json), model weights (*.pt), cache
├── Dockerfile             # Container setup for production API deployment
├── project.md             # Detailed project plan, metrics, & ablation study comparison
├── README.md              # Primary project documentation and quickstart guide
├── render.yaml            # Render deployment configuration blueprint
└── requirements.txt       # Project dependencies
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
