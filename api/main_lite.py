"""FastAPI lightweight application for the Contract Risk Detector (Mini-Transformer only)."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analyze_mini, health
from api.services.mini_predictor import MiniTransformerPredictor

logger = logging.getLogger("api.main_lite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load mini-predictor at startup; release resources on shutdown."""
    
    app.state.mini_predictor = None
    try:
        logger.info("Loading MiniTransformerPredictor from mini_transformer/best_model_v2 …")
        app.state.mini_predictor = MiniTransformerPredictor(
            model_path="mini_transformer/best_model_v2"
        )
        logger.info("MiniTransformerPredictor loaded successfully.")
    except Exception as exc:
        logger.error(
            "MiniTransformerPredictor failed to load. Error: %s",
            exc,
        )
        raise exc

    yield  # app runs here

    logger.info("Shutting down — releasing predictor resources.")
    if hasattr(app.state, "mini_predictor"):
        del app.state.mini_predictor


app = FastAPI(
    title="Contract Risk Detector API (Lite)",
    description="Analyze contract clauses for risk levels using a from-scratch mini-transformer model.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow all origins for development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze_mini.router)
