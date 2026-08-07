"""Health check endpoint for the Contract Risk Detector API."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request):
    """Return API health and the real predictor loading states."""
    legal_predictor = getattr(request.app.state, "predictor", None)
    mini_predictor = getattr(request.app.state, "mini_predictor", None)
    legal_loaded = legal_predictor is not None
    mini_loaded = mini_predictor is not None

    if legal_loaded:
        device = str(legal_predictor.device)
    elif mini_loaded:
        device = str(mini_predictor.device)
    else:
        device = "N/A"

    return {
        "status": "ok",
        "legal_bert_loaded": legal_loaded,
        "mini_transformer_loaded": mini_loaded,
        "device": device,
    }
