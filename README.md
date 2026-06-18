# Crop Disease Detection — Vision Transformer + FastAPI

A fine-tuned Vision Transformer (ViT-Base) that classifies crop leaf images into 38 disease/healthy categories across 14 crop species, served through a FastAPI backend with GradCAM-based explainability.

Trained on PlantVillage, deployed with a real inference API (not just a notebook demo), and evaluated honestly — including where it doesn't generalize well.

## What this actually is

A backend service (`/predict` endpoint) that takes a leaf photo and returns a disease classification, a confidence score, and a visual heatmap showing which part of the image the model used to make its decision. A Gradio frontend sits on top calling that same endpoint over HTTP, the way any real client (mobile app, web app) would — it doesn't touch the model directly.

## Results

- 99.68% accuracy on a held-out test set of 8,147 images (never seen during training)
- 99.73% validation accuracy at the saved checkpoint (epoch 4 of 5)
- Sub-200ms inference latency measured end-to-end through the API (~100ms typical)

## Why those numbers need a caveat

PlantVillage is a lab-controlled dataset — same backgrounds, same lighting, same camera setup per class. Published critiques of this dataset note leaf-grouping leakage: the same physical leaf, photographed multiple times, can end up split across train and test by a random shuffle. High accuracy here is real, but it's a benchmark number, not a field performance number.

I tested this directly. A real-world leaf photo with insect damage and irregular discoloration — visually different from PlantVillage's clean, single-symptom images — got classified at 30.2% confidence, barely above the 2.6% baseline for random guessing across 38 classes. The model didn't confidently hallucinate a wrong answer; it correctly signaled uncertainty on an out-of-distribution image. That's a more honest result than a clean 99% number with no real-world test behind it.

## Explainability — GradCAM, and what it actually showed

I ran GradCAM across batches of 15 and 40 disease-class test images and measured what fraction of the model's strongest attention landed on the leaf itself versus the background (using a simple greenness heuristic — not true segmentation, treat the percentages as directional).

Results across both batches were consistent: mean on-leaf attention sat around 50-54%, with high variance by disease type. Diseases with large, well-defined lesions (Target Spot, Spider Mites) showed strong on-leaf attention (80-90%+). Diseases presenting as small or scattered marks — Bacterial Spot (n=7, 29.5%), Late Blight (n=4, 34.2%), Powdery Mildew (n=3, 19.4%) — consistently showed the model's attention landing mostly on background.

Conclusion: the model isn't purely exploiting background shortcuts, but for several disease types it's likely relying partly on leaf silhouette or holistic image statistics rather than fine-grained symptom localization. This is a known risk with single-background datasets like PlantVillage, and it's documented here rather than hidden behind a high accuracy number.

## Architecture

```
Leaf image upload
       ↓
FastAPI /predict endpoint
       ↓
ViT-Base (fine-tuned, 85.8M params) → class prediction + confidence
       ↓
GradCAM → attention heatmap over the last transformer layer
       ↓
JSON response: crop, condition, confidence, inference time, heatmap (base64)
       ↓
Gradio frontend (calls the API over HTTP, displays result)
```

## Stack

- **Model**: `google/vit-base-patch16-224`, fine-tuned (classifier head replaced: 768 → 38 classes)
- **Training**: PyTorch, AdamW, cosine annealing LR schedule, mixed precision (AMP), 5 epochs
- **Explainability**: `pytorch-grad-cam`, custom reshape transform for ViT's patch-sequence output
- **Serving**: FastAPI + Uvicorn
- **Frontend**: Gradio, calling the API over HTTP rather than the model directly
- **Dataset**: PlantVillage (54,305 images, 38 classes), loaded via HuggingFace Datasets

## Running it

This currently runs inside a single Colab notebook — FastAPI in a background thread, Gradio in the main thread, communicating over `localhost`. Open `crop_disease_fastapi_gradio.ipynb`, run all cells. The notebook loads a pretrained checkpoint rather than retraining (training takes ~40 minutes on a T4 and is included but commented out — the checkpoint is loaded directly).

The Gradio cell prints a public URL valid for 7 days. For permanent hosting, see the HuggingFace Spaces version of this project (linked separately).

### API directly, no UI

```python
import requests

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("leaf.jpg", f, "image/jpeg")}
    )
print(response.json())
```

Returns:
```json
{
  "crop": "Tomato",
  "condition": "Early blight",
  "confidence": 0.987,
  "is_healthy": false,
  "inference_time_ms": 98.3,
  "gradcam_image_base64": "..."
}
```

## What's in the notebook

Dataset loading and exploration, memory-safe preprocessing pipeline (lazy-loaded via HuggingFace's `.select()` rather than materializing 54K images into RAM), model loading and training config, checkpoint loading, full test-set evaluation with confusion matrix and per-class precision/recall, GradCAM setup and the explainability analysis described above, the FastAPI service, and the Gradio frontend.

Every cell has a markdown explanation above it covering what it does and why — built as a teaching reference as much as a working pipeline.

## Limitations, stated plainly

PlantVillage's controlled photography conditions mean this accuracy number won't transfer cleanly to farmer-captured field photos with natural backgrounds and variable lighting. GradCAM analysis shows inconsistent symptom localization for diseases with small or scattered visual markers. The on-leaf attention measurement uses a crude color heuristic, not real segmentation. No field validation has been done yet — that's the obvious next step before this is anything more than a strong benchmark result.

## Background

B.Agric (Hons.), Crop Production and Protection, Obafemi Awolowo University. This is part of a broader effort to build practical AI tooling for smallholder agriculture in sub-Saharan Africa — paired with a RAG-based agronomic advisory system and a commodity price forecasting model in the same portfolio.

---

**Praise Omoolorun Adetifa**
[GitHub](https://github.com/PraiseO2) · [LinkedIn](https://linkedin.com/in/praise-adetifa-3732161b2) · adetifapraisea1@gmail.com
