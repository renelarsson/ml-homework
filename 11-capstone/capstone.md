# Capstone: Computer Vision with TensorFlow

This document proposes 7 computer vision capstone project options, then outlines a standard workflow, architecture, and procedures you can follow to complete any of them in 2–3 weeks using TensorFlow/Keras. AWS Lambda is explicitly excluded; we’ll use FastAPI or ONNX Runtime for serving.

## Project Options (pick one)

1. Composite Sketch Generator (Law Enforcement Aid)
- Goal: Given a few unclear facial images (low resolution, occlusions), generate cleaner composite images and alternative angles/views to aid identification.
- Approach: Image-to-image enhancement + view synthesis.
- Model: Two-stage pipeline — (a) super-resolution/denoising (e.g., ESRGAN-like or SR3-style with TensorFlow), then (b) pose-conditioned face view synthesis (e.g., conditional GAN). Ethics guardrails and dataset constraints required.

2. Face Anti-Spoofing / Deepfake-Resistant Access Control
- Goal: Detect whether a face in an input frame is genuine vs. synthetic/embedded (cyber security fraud guard against unlawful entry).
- Approach: Binary classification with liveness cues (texture cues, moiré patterns, blinking/micro-movements), optionally frequency-domain features.
- Model: CNN backbone (EfficientNetLite) + attention; train with genuine vs. spoof datasets.

3. PPE Compliance Detection (Safety)
- Goal: Detect if people in a scene wear required PPE (helmet, vest, gloves) — a practical CV task.
- Approach: Object detection + person association.
- Model: TensorFlow Object Detection API (SSD-MobileNet) or KerasCV YOLO-like; small dataset + augmentation.

4. Document OCR + Layout Understanding
- Goal: Extract text + layout (titles, tables, figures) for semi-structured documents.
- Approach: Text detection + recognition pipeline.
- Model: EAST/CRAFT-like detection + CRNN recognition in TensorFlow; Tesseract as baseline.

5. Retail Shelf Compliance
- Goal: Detect product presence/placement (planogram compliance) from shelf images.
- Approach: Detection + simple matching rules.
- Model: SSD-MobileNet, KerasCV detections; hard-negative mining.

6. Road Damage Detection (Smart Cities)
- Goal: Classify and localize potholes/cracks in road images.
- Approach: Segmentation or detection.
- Model: U-Net for semantic segmentation; lightweight backbone for edge deployment.

7. Medical X-ray Anomaly Pre-Screen (Education/Practice)
- Goal: Classify suspicious findings (e.g., pneumonia indicators) in chest X-rays.
- Approach: Transfer learning with careful validation; strong disclaimers.
- Model: EfficientNet or ResNet; Grad-CAM for explainability.

Notes:
- 1 and 2 align with your interest in fraud/security. #2 is the closest CV equivalent to your midterm’s fraud detection.
- All projects are feasible in 2–3 weeks with scoped MVP (quality depends on dataset availability).

## Standard 2–3 Week Plan

- Week 1: Scope + Data
  - Define success metrics (e.g., F1 for classification; mAP for detection; PSNR/SSIM for enhancement).
  - Acquire dataset (public sources listed below) and create `data/` with train/val/test splits.
  - Build `notebooks/` exploration: class balance, augmentations.

- Week 2: Modeling
  - Implement TensorFlow model(s) with Keras.
  - Train baseline; add augmentation and regularization.
  - Evaluate; add callbacks (EarlyStopping, ReduceLROnPlateau).

- Week 3: Serving + Packaging
  - Export model (`SavedModel` or `.h5`).
  - Build FastAPI service for local inference; optionally convert to ONNX for CPU speed.
  - Package Docker image; add simple client scripts.

## Repository Layout (following this repo’s conventions)

```
11-capstone/
  capstone.md                # This plan
  data/                      # Place datasets (git-ignored)
  notebooks/
    01-exploration.ipynb
    02-train.ipynb
    03-eval-export.ipynb
  service/
    app.py                   # Container entrypoint (uvicorn)
    predict.py               # Dev entrypoint with schemas
    Dockerfile               # python:3.13-slim + uv
    pyproject.toml           # deps (tensorflow, keras, fastapi, uvicorn, numpy, pillow, opencv-python)
    test.py                  # HTTP probe
  README.md                  # Usage
```

## TensorFlow/Keras Workflow

1) Environment
- Python >= 3.12. Create venv with `uv venv`; install deps via `uv pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow fastapi uvicorn onnxruntime`.

2) Data
- Create `data/` with `train/`, `val/`, `test/`. Use `tf.data` pipelines.
- Apply augmentations via `tf.image` or `keras.preprocessing`.

3) Modeling
- Choose backbone (EfficientNet, MobileNet, ResNet) via `tf.keras.applications`.
- Heads:
  - Classification: `Dense` with softmax/sigmoid.
  - Detection: KerasCV or TensorFlow Object Detection API (if time permits, prefer KerasCV for simplicity).
  - Enhancement: U-Net / SR models.
- Losses/metrics: `BinaryCrossentropy`, `CategoricalCrossentropy`, `Dice/Focal` for segmentation; `MeanIoU`, `AUC`, domain-specific metrics.

4) Training
- Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `TensorBoard`.
- Logging: Save `history.json`, plots to `notebooks/` or `reports/`.

5) Export
- Save as `SavedModel` or `.h5`. Optionally export ONNX: `tf2onnx.convert`.

6) Serving (no AWS Lambda)
- FastAPI (`predict.py`): define Pydantic input, read image, preprocess, run model, return prediction JSON.
- Containerize (`Dockerfile`): run `uvicorn predict:app --host 0.0.0.0 --port 9696`.

## FastAPI Service Skeleton (example)

```python
# service/predict.py
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("model.h5")

IMG_SIZE = (224, 224)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    x = preprocess(img)
    y = model.predict(x)
    return {"scores": y[0].tolist()}
```

## Datasets (per project)

- Composite Sketch Generator: CelebA, LFW (for faces); consider synthetic degradations and pose annotations.
- Face Anti-Spoofing: CelebA-Spoof, SiW, MSU MFSD.
- PPE Detection: PPE datasets on Roboflow; or custom labeling with LabelImg.
- Document OCR: ICDAR datasets; SynthText.
- Retail Shelf: SKU110K; GroZi-120.
- Road Damage: Road Damage Detection datasets (RDD), DeepCrack.
- Medical X-ray: ChestX-ray14, COVIDx (educational use, strict disclaimers).

## Evaluation Targets

- Classification: Accuracy, F1, ROC-AUC; confusion matrix.
- Detection: mAP@[IoU], precision/recall.
- Segmentation: mIoU, Dice.
- Enhancement: PSNR, SSIM.

## Deployment Options (non-AWS)

- Local FastAPI (Dockerized) — primary.
- Optional: ONNX Runtime for CPU inference.
- Optional: Kubernetes (see `10-kubernetes/` examples).

## Next Steps

1. Pick a project (recommend #2 Face Anti-Spoofing for security alignment).
2. I’ll scaffold `11-capstone/service/` with FastAPI + Dockerfile and a training notebook.
3. We’ll acquire a small dataset and start Week 1 tasks.
