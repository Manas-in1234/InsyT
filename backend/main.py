from __future__ import annotations

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .ai_service import interpret
from .analyzer import analyze, read_dataset
from .report import render_report
from .report import render_ml_section
from .ml_service import parse_features, train_regressor

load_dotenv(Path(__file__).with_name(".env"))
MAX_UPLOAD = 10 * 1024 * 1024
app = FastAPI(title="InsyT API", version="1.0.0")
origins = [origin.strip().rstrip("/") for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "InsyT"}


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(400, "Unsupported file type. Upload a CSV or XLSX file.")
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "File exceeds the 10 MB upload limit.")
    try:
        frame = read_dataset(content, filename)
        result = analyze(frame, filename)
        result["metadata"]["size_bytes"] = len(content)
        summary, recommendations, status = interpret(result)
        result["executive_summary"] = summary
        result["recommendations"] = recommendations
        result["ai_status"] = status
        result["report"] = render_report(result)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # Avoid disclosing uploaded content, paths, internals, or environment.
        logging.exception("Analysis failed")
        raise HTTPException(500, "Analysis could not be completed. Please check the file and try again.") from exc


@app.post("/api/ml/train")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...),
                      feature_columns: str = Form(...), test_size: float = Form(0.2),
                      n_estimators: int = Form(100)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(400, "Unsupported file type. Upload a CSV or XLSX file.")
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "File exceeds the 10 MB upload limit.")
    try:
        frame = read_dataset(content, filename)
        features = parse_features(feature_columns)
        result = train_regressor(frame, target_column, features, test_size, n_estimators, filename)
        statistics = analyze(frame.copy(), filename)
        statistics["metadata"]["size_bytes"] = len(content)
        summary, recommendations, status = interpret(statistics)
        statistics["executive_summary"] = summary
        statistics["recommendations"] = recommendations
        statistics["ai_status"] = status
        result["report"] = render_report(statistics) + "\n" + render_ml_section(result)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logging.exception("Model training failed")
        raise HTTPException(500, "Model training could not be completed. Please check the selected columns and try again.") from exc
