"""
api.py
────────────────────────────────────────────────────────────────────────────────
FastAPI REST backend – optional companion to the Gradio UI.

Run standalone:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints
---------
POST /generate          – generate filter code from prompt
POST /apply             – apply filter to uploaded image  (multipart/form-data)
GET  /filters           – list saved filters
POST /filters           – save a filter
DELETE /filters/{name}  – delete a filter
POST /filters/{name}/share  – share a filter via webhook
GET  /health            – health check
"""

from __future__ import annotations

import base64
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_generator import AIFilterGenerator
from validator    import FilterValidator
from compiler     import FilterCompiler
from database     import FilterDatabase
from share_service import ShareService


app = FastAPI(title="AI Filter Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module instances
validator = FilterValidator()
compiler  = FilterCompiler()
generator = AIFilterGenerator(validator=validator, compiler=compiler)
db        = FilterDatabase()
share_svc = ShareService()


# ── Request / response models ─────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    groq_api_key: Optional[str] = ""
    openrouter_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""


class GenerationAttemptRecord(BaseModel):
    stage: str
    provider: str
    model: str
    success: bool
    code: str = ""
    error: str = ""
    validation_message: str = ""
    latency_ms: int = 0

class GenerateResponse(BaseModel):
    session_id: str
    code:    str
    valid:   bool
    message: str
    final_stage: str
    attempts: List[GenerationAttemptRecord]

class SaveFilterRequest(BaseModel):
    name: str
    code: str

class FilterRecord(BaseModel):
    name:       str
    code:       str
    created_at: str
    updated_at: Optional[str] = None

class ApplyResponse(BaseModel):
    image_b64: str    # base64-encoded PNG
    message:   str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        db.ping()
        mongo_status = "ok"
    except Exception as exc:
        mongo_status = f"error: {exc}"
    return {"status": "ok", "mongodb": mongo_status}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate filter code from a natural-language prompt."""
    try:
        result = generator.generate(
            prompt=req.prompt,
            groq_api_key=req.groq_api_key,
            openrouter_api_key=req.openrouter_api_key,
            gemini_api_key=req.gemini_api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        db.save_generation_session(result.to_dict())
    except Exception:
        pass

    return GenerateResponse(
        session_id=result.session_id,
        code=result.code,
        valid=result.success,
        message=result.message,
        final_stage=result.final_stage,
        attempts=result.attempts,
    )


@app.post("/apply", response_model=ApplyResponse)
async def apply_filter(
    code:  str         = Form(...),
    image: UploadFile  = File(...),
):
    """Apply a filter to an uploaded image. Returns base64-encoded PNG."""
    # Validate + compile
    ok, msg = validator.validate(code)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Validation failed: {msg}")

    fn, err = compiler.compile_and_smoke_test(code)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"Compile error: {err}")

    # Decode uploaded image
    data  = await image.read()
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        result = fn(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter execution error: {e}")

    # Encode result as PNG
    _, buf = cv2.imencode(".png", result)
    b64 = base64.b64encode(buf).decode()
    return ApplyResponse(image_b64=b64, message="Filter applied.")


@app.get("/filters", response_model=List[FilterRecord])
def list_filters():
    try:
        return db.get_all_filters()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@app.post("/filters", status_code=201)
def save_filter(req: SaveFilterRequest):
    ok, msg = validator.validate(req.code)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid filter code: {msg}")
    try:
        db.save_filter(req.name, req.code)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    return {"message": f"Filter '{req.name}' saved."}


@app.delete("/filters/{name}")
def delete_filter(name: str):
    try:
        db.delete_filter(name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    return {"message": f"Filter '{name}' deleted."}


@app.post("/filters/{name}/share")
def share_filter(name: str, preview_b64: str = ""):
    try:
        code = db.get_filter_code(name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    if code is None:
        raise HTTPException(status_code=404, detail=f"Filter '{name}' not found.")
    ok, msg = share_svc.share(name, code, preview_b64)
    if not ok:
        raise HTTPException(status_code=502, detail=msg)
    return {"message": msg}
