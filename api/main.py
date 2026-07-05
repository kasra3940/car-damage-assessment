"""
FastAPI endpoint for the car damage assessment pipeline.

Run with:
    uvicorn api.main:app --reload
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query

from src.pipeline import run_pipeline

app = FastAPI(
    title="Car Damage Assessment API",
    description="Upload a photo of a damaged car and get a structured, "
                 "explainable damage assessment report.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess")
async def assess_damage(
    file: UploadFile = File(...),
    region: str = Query(default="US", description="Pricing region code, e.g. US, IR"),
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / file.filename
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_pipeline(str(image_path), region=region)

    return result
