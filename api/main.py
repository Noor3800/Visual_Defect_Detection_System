from io import BytesIO
from pathlib import Path
import logging

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from model import create_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("defect-detection")

app = FastAPI(
    title="Visual Defect Detection API",
    version="1.0.0",
    description="Manufacturing product normal/defective image classifier.",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The order must match ImageFolder/model class indices after the dataset is received.
CLASS_NAMES = ["defective", "normal"]

MODEL_PATH = Path("models/best_model.pth")

model = None

if MODEL_PATH.exists():
    model = create_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image.",
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Train the model first.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        image = Image.open(BytesIO(contents)).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_value = float(confidence.item())

        logger.info(
            "prediction=%s confidence=%.4f",
            predicted_class,
            confidence_value,
        )

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence_value, 4),
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process image.",
        )
