from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)
from fastapi.middleware.cors import CORSMiddleware

from inference import MRIModelService


app = FastAPI(
    title="NeuroInsight-PD MRI API",
    version="1.0.0",
    description=(
        "Brain MRI Parkinson classification API."
    )
)

# Allow the doctor/radiologist web portal (and local dev builds of it) to
# call this API from the browser. Without this, every request is blocked
# by the browser's CORS policy before it even reaches our routes below.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = MRIModelService()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png"
}

MAX_FILE_SIZE = 10 * 1024 * 1024


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "MRI EfficientNetB0",
        "version": "1.0.0"
    }


@app.post("/predict/mri")
async def predict_mri(
    file: UploadFile = File(...)
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB."
        )

    try:
        result = model_service.predict(
            image_bytes
        )

        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be processed."
        )
