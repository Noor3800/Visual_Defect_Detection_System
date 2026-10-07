from io import BytesIO
from pathlib import Path
import logging
import sys
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from model import create_model


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("defect-detection")


# FastAPI application
app = FastAPI(
    title="Visual Defect Detection API",
    version="1.0.0",
    description=(
        "Manufacturing product image classifier "
        "for Normal vs Defective detection."
    ),
)


# Device

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Class mapping

CLASS_NAMES = [
    "defective",
    "normal",
]


# Model

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

model = None

if MODEL_PATH.exists():

    try:

        logger.info(
            "Loading model from: %s",
            MODEL_PATH,
        )

        model = create_model(
            num_classes=len(CLASS_NAMES)
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        # Support both a raw state_dict and a checkpoint
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            checkpoint = checkpoint["model_state_dict"]

        model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()

        logger.info(
            "Model loaded successfully."
        )

        logger.info(
            "Using device: %s",
            DEVICE,
        )

    except Exception:

        logger.exception(
            "Failed to load trained model."
        )

        model = None

else:

    logger.warning(
        "Model file not found: %s",
        MODEL_PATH,
    )


# Image preprocessing

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# Health endpoint

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
    }


# Prediction endpoint

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Validate content type

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload a JPG, JPEG, PNG, "
                "or WEBP image."
            ),
        )


    # Check model

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not available. "
                "Train the model first."
            ),
        )
    # Read uploaded file

    try:

        contents = await file.read()

        # Maximum image size = 10 MB
        max_file_size = 10 * 1024 * 1024

        if len(contents) > max_file_size:

            raise HTTPException(
                status_code=413,
                detail=(
                    "Image is too large. "
                    "Maximum allowed size is 10 MB."
                ),
            )

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # Open image

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except HTTPException:
        raise

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    except Exception:

        logger.exception(
            "Failed to read uploaded image."
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image.",
        )


    # Preprocess image

    try:

        tensor = TRANSFORM(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(DEVICE)

    except Exception:

        logger.exception(
            "Image preprocessing failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Image preprocessing failed.",
        )

    # Model inference
    start_time = time.perf_counter()

    try:

        with torch.no_grad():

            logits = model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1,
            )

    except Exception:

        logger.exception(
            "Model inference failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Model inference failed.",
        )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Convert prediction
    predicted_index = predicted.item()

    confidence_value = (
        confidence.item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Logging
    logger.info(
        "prediction=%s confidence=%.4f "
        "latency=%.2fms file=%s",
        predicted_class,
        confidence_value,
        inference_time_ms,
        file.filename,
    )

    # Response

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(
            confidence_value,
            4,
        ),
        "inference_time_ms": round(
            inference_time_ms,
            2,
        ),
    }