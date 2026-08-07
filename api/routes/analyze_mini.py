"""Mini-transformer analysis endpoints."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/analyze", tags=["Analysis"])


class ClauseRequest(BaseModel):
    """Request body for single-clause analysis."""

    text: str


@router.post("/clause/mini")
async def analyze_clause_mini(body: ClauseRequest, request: Request):
    """Analyze a single text clause with the mini-transformer model."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text field is empty after stripping whitespace.")

    predictor = getattr(request.app.state, "mini_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Mini-transformer predictor is unavailable on this deployment.",
        )

    try:
        result = predictor.predict(text)
        reason = predictor.get_risk_reason(result["risk_label"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return {
        "clause_text": text,
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "risk_reason": reason,
        "model": "mini_transformer_from_scratch",
    }
