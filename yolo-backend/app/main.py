from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from app.utils import decode_image
from app.predict import run_prediction

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
    """Health check — visit this in browser to confirm API is running."""
    return {"status": "running", "message": "Tea Detection API is live"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a tea image and get back:
    - annotated_image: original photo with bounding boxes (base64)
    - detections: list of detected shoots with class + confidence
    - summary: pluckable count, top class, recommendation
    """

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    start = time.time()

    # Read and decode image
    contents = await file.read()
    try:
        img = decode_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run model
    result = run_prediction(img)

    # Add timing
    result["inference_ms"] = round((time.time() - start) * 1000, 1)
    result["filename"]     = file.filename

    return result