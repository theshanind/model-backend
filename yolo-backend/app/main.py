from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from app.utils import decode_image
from app.predict import run_prediction
from app.predict_cls import run_classification

app = FastAPI(
    title="Tea Flush Detection API",
    description="Detects and classifies tea shoot flush types using YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status" : "running",
        "message": "Tea Detection API is live",
        "models" : {
            "detection"     : "ready",
            "classification": "ready",   # update to 'ready' when model 2 is added
        }
    }


# ── Model 1: unchanged — frontend keeps working exactly as before ──────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    start    = time.time()
    contents = await file.read()

    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = run_prediction(img)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result["inference_ms"] = round((time.time() - start) * 1000, 1)
    result["filename"]     = file.filename
    return result


# ── Model 2: stub route — returns 501 until you implement it ───────────────
@app.post("/predict/classify")
async def predict_classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )
    
    start    = time.time()
    contents = await file.read()
 
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    try:
        result = run_classification(img)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
 
    result["inference_ms"] = round((time.time() - start) * 1000, 1)
    result["filename"]     = file.filename
    return result