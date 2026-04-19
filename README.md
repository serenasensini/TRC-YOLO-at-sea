# 🚢 YOLO at Sea — Real-Time AI for Ship and Wreck Detection

> **Talk Abstract:** *The ocean hides stories—some sailing proudly, others resting as silent wrecks beneath the waves. This project dives into how YOLO, a state-of-the-art object detection model, can revolutionize maritime monitoring by identifying ships and wrecks in real time from satellite and aerial imagery.*

## 🏗️ Project Structure

```
TheRedCode-YOLO-at-sea/
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Dependencies
├── configs/
│   ├── dataset.yaml            # YOLO dataset configuration
│   └── wreck_coordinates.csv   # Google Maps wreck pins
├── scripts/
│   ├── download_kaggle.py      # Step 1: Download Kaggle dataset
│   ├── download_satellite.py   # Step 2: Download satellite images
│   ├── auto_annotate.py        # Step 3: Auto-annotation (GDINO+SAM2)
│   ├── prepare_dataset.py      # Step 4: Dataset preparation
│   ├── train.py                # Step 5: YOLOv11 training
│   └── stream_demo.py          # Step 6: Real-time streaming demo
├── dataset/
│   ├── kaggle_raw/             # Ships in Satellite Imagery (ship/no_ship)
│   ├── satellite_raw/          # Downloaded satellite images
│   ├── labels_auto/            # Auto-generated annotations
│   ├── images/{train,val}/     # Final dataset images
│   └── labels/{train,val}/     # Final dataset labels
├── models/                     # Trained weights (best.pt)
└── runs/                       # Training logs & demo output
```

## 🚀 Quick Start

### Prerequisites (manual steps)

1. **Python 3.10+** with a virtual environment
2. **Kaggle API Token**: Download from [kaggle.com/settings](https://www.kaggle.com/settings) → place in `~/.kaggle/kaggle.json`
3. **Satellite Imagery API Key**: Either [Google Static Maps](https://console.cloud.google.com/apis/credentials) or [Mapbox](https://account.mapbox.com/access-tokens/) (recommended if Google satellite is unavailable in your region)
4. **GPU (recommended)**: NVIDIA GPU with ≥8GB VRAM for training

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Check pipeline status
python main.py --status
```

### Run the Pipeline

```bash
# Step by step
python scripts/download_kaggle.py
python scripts/download_satellite.py --provider mapbox --api-key YOUR_MAPBOX_TOKEN
python scripts/auto_annotate.py
python scripts/prepare_dataset.py
python scripts/train.py --epochs 100 --device 0
python scripts/stream_demo.py

# Or all at once
python main.py --step all
```

## 📋 Pipeline Steps

### Step 1 — Download Kaggle Dataset
```bash
python scripts/download_kaggle.py
```
Downloads and extracts the [Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) dataset from Kaggle. Contains ~4000 satellite image chips (80×80 px) from Planet Labs, each classified as "ship" or "no-ship". Images are organized into `ship/` and `no_ship/` subdirectories.

### Step 2 — Download Satellite Images
```bash
# Google (default)
python scripts/download_satellite.py --api-key YOUR_GOOGLE_API_KEY
# Mapbox (if Google satellite is unavailable in your region)
python scripts/download_satellite.py --provider mapbox --api-key YOUR_MAPBOX_TOKEN
```
Downloads high-resolution satellite images from Google Static Maps or Mapbox for each wreck coordinate in `configs/wreck_coordinates.csv`. Images are captured at zoom levels 17-19 (~0.3-1.2m/px).

**💡 Edit `configs/wreck_coordinates.csv` to add your own Google Maps pins!**

### Step 3 — Auto-Annotation + Label Studio Review
```bash
python scripts/auto_annotate.py
```
Automatically generates initial YOLO-format bounding box annotations:
- **Kaggle images**: Labels derived from the ship/no-ship classification.
- **Satellite images**: Uses Grounding DINO + SAM2 (zero-shot), with YOLOv8 COCO fallback.

#### Label Studio Review (required)

After auto-annotation, review and correct labels using [Label Studio](https://labelstud.io/):

1. **Install & launch Label Studio:**
   ```bash
   pip install label-studio
   label-studio start
   ```
2. **Create a new project** and import your images from `dataset/kaggle_raw/` and `dataset/satellite_raw/`.
3. **Configure the labeling interface** — use the **Object Detection with Bounding Boxes** template and set these labels:
   - `ship` (class 0) — Active vessels
   - `wreck` (class 1) — Shipwrecks, beached/rusted ships
   - `sea` (class 2) — Open sea / background water areas
4. **Pre-import auto-generated labels** (optional): upload the `.txt` files from `dataset/labels_auto/` to speed up review.
5. **Label/correct all images** in the Label Studio UI.
6. **Export in YOLO format**: go to *Export* → select **YOLO** format. This produces a zip with a `labels/` folder.
7. **Copy the exported labels** into the project:
   ```bash
   # Unzip the Label Studio export and copy label .txt files
   cp /path/to/label-studio-export/labels/*.txt dataset/labels_studio/
   ```
   The `dataset/labels_studio/` folder takes priority over `dataset/labels_auto/` during dataset preparation.

> **⚠️ Important:** Make sure Label Studio's YOLO export maps classes in this order: `0=ship`, `1=wreck`, `2=sea`. Verify by checking a few exported `.txt` files.

### Step 4 — Prepare Dataset
```bash
python scripts/prepare_dataset.py
```
Merges all images and labels (preferring Label Studio labels over auto-generated ones), applies 80/20 train/val split, resizes to 640×640, and generates augmented variants (flip, rotation, brightness).

The script looks for labels in this order of priority:
1. `dataset/labels_studio/` — Label Studio corrected labels ✅
2. `dataset/labels_auto/` — Auto-generated labels (fallback)

### Step 5 — Train YOLOv11
```bash
# With GPU (recommended, requires ≥8GB VRAM)
python scripts/train.py --epochs 100 --device 0

# CPU only (much slower)
python scripts/train.py --epochs 100 --device cpu
```
Fine-tunes YOLOv11m on the prepared dataset. The model will learn to detect 3 classes: **ship**, **wreck**, and **sea**. Key params:
- Mosaic + MixUp augmentation
- AdamW optimizer with warmup
- Early stopping (patience=20)

After training, the best weights are saved to `models/best.pt` and validation metrics (mAP50, mAP50-95) are printed.

> **💡 Tip:** If training loss plateaus early, try increasing epochs or reducing batch size (`--batch 8`).

### Step 6 — Real-Time Streaming Demo
```bash
python scripts/stream_demo.py
```
Simulates a maritime surveillance system by processing images one-by-one with live visualization:
- Bounding boxes color-coded by class: 🟢 ship, 🔴 wreck, 🔵 sea
- Side panel with live statistics
- Video recording (MP4) saved to `runs/demo_output.mp4`

**Controls:** `q` = quit | `n` = next | `p` = pause

If no trained model is found in `models/best.pt`, the demo falls back to a pre-trained YOLOv8n (COCO) model.

## 🛰️ Image Sources

| Source | Resolution | Cost | Verdict |
|--------|-----------|------|---------|
| **Kaggle Ships in Satellite Imagery** | ~3m/px (80×80 crops) | Free | ✅ Ship/no-ship classification |
| **Google Static Maps** | ~0.3m/px (z19) | Free tier: 28k/month | ✅ High-res (if available) |
| **Mapbox Satellite** | ~0.5m/px | Free tier: 50k/month | ✅ Recommended alternative |
| Sentinel-2 (Copernicus) | 10m/px | Free | ❌ Too coarse for wrecks |
| Planet Labs | 3m/px | Commercial | For production |

The Kaggle dataset provides a large volume of pre-labeled ship/no-ship satellite crops from Planet Labs. Google Static Maps or Mapbox at zoom 18-19 gives sub-meter resolution, ideal for detecting ships and wrecks in larger scenes.

## 🏷️ Auto-Annotation Strategy

The annotation pipeline uses a **two-source approach**:

1. **Kaggle ship/no-ship images** (80×80 crops): Labels are derived directly from the dataset classification. Ship images receive a centered bounding box; no-ship images serve as background (negative examples).

2. **Satellite images** (Google/Mapbox, 640×640):
   - **Grounding DINO + SAM2** (zero-shot): Uses text prompts ("ship", "wreck", "vessel") to detect and segment objects.
   - **YOLOv8 COCO fallback**: Uses pre-trained COCO model to detect `boat` class.

**Manual review is always recommended** for a subset of annotations to ensure quality.

## 📊 Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `ship` | Active vessels (cargo, tanker, fishing, etc.) |
| 1 | `wreck` | Shipwrecks, beached/rusted ships, sunken structures |
| 2 | `sea` | Open sea, background water areas |

## 📄 License

This project is for educational/research purposes. Satellite images are subject to their respective providers' Terms of Service (Google, Mapbox, Planet Labs).

