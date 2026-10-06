# Visual_Defect_Detection_System
# Computer Vision Defect Detection

Production-oriented visual defect detection system for a manufacturing quality-inspection scenario.

## Assessment alignment

The SparkAI assessment asks for:
- dataset analysis and preparation
- binary Normal vs Defective classification
- preprocessing and augmentation
- class-imbalance handling
- train/validation/test evaluation
- precision, recall, F1 and confusion matrix
- false-positive / false-negative analysis
- model serving through an image API
- validation, logging and error handling
- realistic packaging/deployment

## Dataset

This project uses **MVTec AD** as a fallback public industrial-inspection dataset because the assessment PDF states that its dataset would be provided separately.

Official MVTec AD:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

MVTec AD contains 15 object/texture categories and more than 5,000 high-resolution images. Its official setup is an anomaly-detection benchmark: training images are defect-free, while test images contain both defect-free and defective examples.

### Important methodological note

The SparkAI assessment asks for a supervised binary classifier. MVTec AD's official benchmark protocol is not a normal supervised binary-classification split.

Therefore, this project uses a **custom binary-classification protocol** on one MVTec category:

- Category: `bottle`
- `normal`: MVTec `train/good` + `test/good`
- `defective`: all MVTec `test/<defect_type>` images
- These labeled images are then split into train/validation/test using a fixed seed.

This is a custom classification experiment and **must not be presented as the official MVTec AD benchmark evaluation**.

The official MVTec dataset contains 209 training images for Bottle, 20 defect-free test images, and 63 defective test images.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset placement

After downloading and extracting MVTec AD, place it as:

```text
data/
└── raw/
    └── mvtec_ad/
        ├── bottle/
        ├── cable/
        ├── capsule/
        └── ...
```

For this project, only the `bottle` category is required.

Expected structure:

```text
data/raw/mvtec_ad/bottle/
├── train/
│   └── good/
└── test/
    ├── good/
    ├── broken_large/
    ├── broken_small/
    └── contamination/
```

The exact defect folder names come from the downloaded dataset.

## Prepare the binary dataset

Run:

```powershell
python src/prepare_mvtec.py
```

This creates:

```text
data/
├── train/
│   ├── normal/
│   └── defective/
├── val/
│   ├── normal/
│   └── defective/
└── test/
    ├── normal/
    └── defective/
```

## Train

```powershell
python src/train.py
```

The first run may download ImageNet weights for EfficientNet-B0.

The best checkpoint is saved as:

```text
models/best_model.pth
```

## Evaluate

```powershell
python src/evaluate.py
```

Outputs include:

```text
results/classification_report.json
results/confusion_matrix.png
```

## Model choice

EfficientNet-B0 is used with ImageNet transfer learning.

Reasoning:
- small industrial dataset
- strong visual feature extraction from pretrained weights
- relatively lightweight inference
- practical accuracy/latency trade-off
- straightforward PyTorch deployment

The pretrained feature extractor is initially frozen and the classification head is trained for the two classes.

## Class imbalance

Training uses inverse-frequency class weights in CrossEntropyLoss so the model does not treat a larger class as more important simply because it has more images.

The actual class distribution should always be reported rather than assuming the classes are balanced.

## Preprocessing

Training:
- resize to 224×224
- horizontal flip
- small rotation
- mild brightness/contrast variation
- ImageNet normalization

Validation/test:
- resize to 224×224
- ImageNet normalization
- no random augmentation

## Evaluation

The final report will include:
- precision
- recall
- F1-score
- confusion matrix

For manufacturing inspection, **defective recall** is particularly important because a false negative means a defective product was classified as normal.

## API

Run:

```powershell
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Endpoints:

```text
GET  /health
POST /predict
```

Example response:

```json
{
  "predicted_class": "defective",
  "confidence": 0.94
}
```

## Docker

```powershell
docker build -t computer-vision-defect-detection .
docker run -p 8000:8000 computer-vision-defect-detection
```

## Limitations

1. MVTec AD was not the dataset originally promised in the assessment.
2. The custom binary split uses images from MVTec's official test set, so it is not comparable to the official MVTec anomaly-detection benchmark.
3. Only one product category is used.
4. The classifier predicts image-level Normal/Defective status; it does not localize the defect.
5. Real production deployment would require validation on the client's own camera setup, lighting, products and defect distribution.

## If SparkAI later provides its dataset

Replace the MVTec preparation step with the supplied dataset and rerun the dataset analysis. Do not mix the SparkAI dataset with MVTec unless explicitly justified.
