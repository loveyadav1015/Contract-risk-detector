"""Analysis endpoints for the Contract Risk Detector API.

POST /analyze        — Upload a PDF for full contract risk analysis
POST /analyze/clause — Analyze a single clause (raw text)
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.services.extractor import extract_text_from_pdf, split_into_clauses

router = APIRouter(prefix="/analyze", tags=["Analysis"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ClauseRequest(BaseModel):
    """Request body for single-clause analysis."""
    text: str


# ---------------------------------------------------------------------------
# POST /analyze/clause — single clause
# ---------------------------------------------------------------------------


@router.post("/clause")
async def analyze_clause(body: ClauseRequest, request: Request):
    """Analyze a single text clause for risk level.

    Returns the predicted risk level, confidence score (real softmax
    output), and a heuristic risk reason.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text field is empty after stripping whitespace.")

    predictor = request.app.state.predictor

    try:
        result = predictor.predict(text)
        reason = predictor.get_risk_reason(result["risk_label"])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        )

    return {
        "clause_text": text,
        "risk_level": result["risk_level"],
        "confidence": result["confidence"],
        "risk_reason": reason,
    }


# ---------------------------------------------------------------------------
# POST /analyze — full PDF document
# ---------------------------------------------------------------------------


@router.post("")
async def analyze_document(request: Request, file: UploadFile = File(...)):
    """Upload a PDF contract and get per-clause risk analysis.

    Returns a document-level summary and a list of clause-level
    risk assessments.
    """
    # Validate content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Expected content_type 'application/pdf', got '{file.content_type}'.",
        )

    # Extract text from PDF
    try:
        file_bytes = await file.read()
        full_text = extract_text_from_pdf(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"PDF extraction failed: {exc}",
        )

    # Split into clauses
    clauses = split_into_clauses(full_text)
    if not clauses:
        raise HTTPException(
            status_code=422,
            detail="Zero clauses extracted from the PDF after splitting and filtering.",
        )

    # Run inference on each clause
    predictor = request.app.state.predictor
    clause_results = []
    high_count = 0
    medium_count = 0
    low_count = 0

    try:
        for clause_text in clauses:
            result = predictor.predict(clause_text)
            reason = predictor.get_risk_reason(result["risk_label"])
            clause_results.append({
                "clause_text": clause_text,
                "risk_level": result["risk_level"],
                "confidence": result["confidence"],
                "risk_reason": reason,
            })

            if result["risk_label"] == 2:
                high_count += 1
            elif result["risk_label"] == 1:
                medium_count += 1
            else:
                low_count += 1
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error during document analysis: {exc}",
        )

    # Determine overall risk as the highest risk found in any clause
    if high_count > 0:
        overall_risk = "high"
    elif medium_count > 0:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return {
        "document_summary": {
            "overall_risk": overall_risk,
            "high_risk_count": high_count,
            "medium_risk_count": medium_count,
            "low_risk_count": low_count,
            "total_clauses_analyzed": len(clause_results),
        },
        "clauses": clause_results,
    }
