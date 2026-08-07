"""FastAPI application for the Contract Risk Detector."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analyze, analyze_mini, health
from api.services.mini_predictor import MiniTransformerPredictor
from api.services.predictor import RiskPredictor
from src.utils import load_config




logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load predictors at startup; release resources on shutdown."""
    config = load_config()
    model_path: str = config["paths"]["best_model_path"]

    logger.info("Loading RiskPredictor from %s …", model_path)
    predictor = RiskPredictor(model_path=model_path)
    app.state.predictor = predictor
    logger.info("RiskPredictor loaded successfully.")

    app.state.mini_predictor = None
    try:
        logger.info("Loading MiniTransformerPredictor from mini_transformer/best_model_v2 …")
        app.state.mini_predictor = MiniTransformerPredictor(
            model_path="mini_transformer/best_model_v2"
        )
        logger.info("MiniTransformerPredictor loaded successfully.")
    except Exception as exc:
        logger.warning(
            "MiniTransformerPredictor failed to load; continuing with LegalBERT only. Error: %s",
            exc,
        )

    yield  # app runs here

    logger.info("Shutting down — releasing predictor resources.")
    if hasattr(app.state, "predictor"):
        del app.state.predictor
    if hasattr(app.state, "mini_predictor"):
        del app.state.mini_predictor


app = FastAPI(
    title="Contract Risk Detector API",
    description="Analyze contract clauses for risk levels using a fine-tuned LegalBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow all origins for development.
# NOTE: This should be restricted to specific origins before real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(analyze_mini.router)
