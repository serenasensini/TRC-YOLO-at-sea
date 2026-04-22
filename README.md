# 🚢 YOLO at Sea — Real-Time AI for Wreck Detection

> **Talk Abstract:** *The ocean hides stories—some sailing proudly, others resting as silent wrecks beneath the waves. This project dives into how YOLO, a state-of-the-art object detection model, can revolutionize maritime monitoring by identifying wrecks in real time from satellite imagery.*

## 🏗️ Project Structure

```
TheRedCode-YOLO-at-sea/
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Dependencies
├── configs/
│   ├── dataset.yaml            # YOLO dataset configuration (nc=1, Wreck)
│   └── wreck_coordinates.csv   # Google Maps wreck pins
├── scripts/
│   ├── download_satellite.py   # Step 1: Download satellite images
│   ├── auto_annotate.py        # Step 2: Auto-annotation (GDINO+SAM2)
│   ├── prepare_dataset.py      # Step 3: Dataset preparation
│   ├── train.py                # Step 4: YOLOv11 training
│   └── stream_demo.py          # Step 5: Real-time streaming demo
├── dataset/
│   ├── satellite_raw/          # Downloaded satellite images (wrecks)
│   ├── labels_auto/            # Auto-generated annotations
│   ├── labels_studio/          # Label Studio corrected annotations
│   ├── images/{train,val}/     # Final dataset images
│   └── labels/{train,val}/     # Final dataset labels
├── models/                     # Trained weights (best.pt)
├── notebooks/
│   └── train_colab.ipynb       # Google Colab training notebook
└── runs/                       # Training logs & demo output
```

## 🚀 Quick Start

### Prerequisites (manual steps)

1. **Python 3.10+** with a virtual environment
2. **Satellite Imagery API Key**: Either [Google Static Maps](https://console.cloud.google.com/apis/credentials) or [Mapbox](https://account.mapbox.com/access-tokens/) (recommended)
3. **GPU (recommended)**: NVIDIA GPU with ≥8GB VRAM for training, or use [Google Colab](https://colab.research.google.com/) (T4 GPU free tier)

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
python scripts/download_satellite.py --provider mapbox --api-key YOUR_MAPBOX_TOKEN
python scripts/auto_annotate.py
python scripts/prepare_dataset.py
python scripts/train.py --epochs 200 --device 0
python scripts/stream_demo.py

# Or all at once
python main.py --step all
```

## 📋 Pipeline Steps

### Step 1 — Download Satellite Images
```bash
# Google (default)
python scripts/download_satellite.py --api-key YOUR_GOOGLE_API_KEY
# Mapbox (recommended)
python scripts/download_satellite.py --provider mapbox --api-key YOUR_MAPBOX_TOKEN
```
Downloads high-resolution satellite images from Google Static Maps or Mapbox for each wreck coordinate in `configs/wreck_coordinates.csv`. Images are captured at zoom levels 17-19 (~0.3-1.2m/px).

**💡 Edit `configs/wreck_coordinates.csv` to add your own Google Maps pins!**

### Step 2 — Auto-Annotation + Label Studio Review
```bash
python scripts/auto_annotate.py
```
Automatically generates initial YOLO-format bounding box annotations for wreck detection:
- **Grounding DINO + SAM2** (zero-shot): Uses text prompts ("shipwreck", "wreck", "rusted ship", "beached ship", etc.) to detect wrecks.
- **YOLOv8 COCO fallback**: Uses pre-trained COCO model to detect `boat` class as proxy for wrecks.
- All detections are mapped to **class 0 (Wreck)**.

#### Label Studio Review (recommended)

After auto-annotation, review and correct labels using [Label Studio](https://labelstud.io/):

1. **Install & launch Label Studio:**
   ```bash
   pip install label-studio
   label-studio start
   ```
2. **Create a new project** and import your images from `dataset/satellite_raw/`.
3. **Configure the labeling interface** — use the **Object Detection with Bounding Boxes** template with a single label:
   - `Wreck` (class 0) — Shipwrecks, beached/rusted ships, stranded vessels
4. **Pre-import auto-generated labels** (optional): upload the `.txt` files from `dataset/labels_auto/` to speed up review.
5. **Label/correct all images** in the Label Studio UI.
6. **Export in YOLO format**: go to *Export* → select **YOLO** format.
7. **Copy the exported labels** into the project:
   ```bash
   cp /path/to/label-studio-export/labels/*.txt dataset/labels_studio/labels/
   ```
   The `dataset/labels_studio/` folder takes priority over `dataset/labels_auto/` during dataset preparation.

> **⚠️ Important:** Make sure Label Studio's YOLO export maps the class as `0=Wreck`. Verify by checking a few exported `.txt` files — all lines should start with `0`.

### Step 3 — Prepare Dataset
```bash
python scripts/prepare_dataset.py
```
Collects satellite images with their labels (preferring Label Studio over auto-generated), applies 80/20 train/val split, and resizes to 640×640.

The script filters labels to keep **only wreck annotations (class 0)**, automatically handling different labeling schemes from Label Studio and auto-annotate.

### Step 4 — Train YOLOv11
```bash
# With GPU (recommended, requires ≥8GB VRAM)
python scripts/train.py --epochs 200 --device 0

# CPU only (much slower)
python scripts/train.py --epochs 200 --device cpu

# Or use Google Colab (recommended for free GPU)
# Upload dataset.zip + notebooks/train_colab.ipynb to Colab
```
Fine-tunes YOLOv11 on the prepared dataset for **single-class wreck detection**. Key training features:
- `single_cls=True` — treats all annotations as one class
- `freeze=10` — freezes backbone layers to prevent overfitting on small datasets
- Low learning rate (`lr0=0.0005`) for stable fine-tuning
- Aggressive augmentation (mosaic, mixup, copy_paste, rotation, flip)
- AdamW optimizer with dropout regularization
- Early stopping (patience=40)

After training, the best weights are saved to `models/best.pt`.

> **💡 Tip:** For best results, use the Colab notebook (`notebooks/train_colab.ipynb`) with a T4 GPU.

### Step 5 — Real-Time Streaming Demo
```bash
python scripts/stream_demo.py
```
Simulates a maritime surveillance system by processing satellite images one-by-one with live visualization:
- Bounding boxes for detected wrecks (🔴 red)
- Side panel with live statistics
- Video recording (MP4) saved to `runs/demo_output.mp4`

**Controls:** `q` = quit | `n` = next | `p` = pause

If no trained model is found in `models/best.pt`, the demo falls back to a pre-trained YOLOv8n (COCO) model.

## 🛰️ Image Sources

| Source | Resolution | Cost | Verdict |
|--------|-----------|------|---------|
| **Google Static Maps** | ~0.3m/px (z19) | Free tier: 28k/month | ✅ High-res (if available) |
| **Mapbox Satellite** | ~0.5m/px | Free tier: 50k/month | ✅ Recommended |
| Sentinel-2 (Copernicus) | 10m/px | Free | ❌ Too coarse for wrecks |
| Planet Labs | 3m/px | Commercial | For production |

Google Static Maps or Mapbox at zoom 18-19 gives sub-meter resolution, ideal for detecting wrecks in satellite scenes.

## 🏷️ Auto-Annotation Strategy

The annotation pipeline uses **zero-shot detection** on satellite images:

- **Grounding DINO + SAM2** (preferred): Uses text prompts ("shipwreck", "wreck", "rusted ship", "beached ship", "abandoned ship", "stranded vessel") to detect and segment wreck objects.
- **YOLOv8 COCO fallback**: Uses pre-trained COCO model to detect `boat` class as a proxy for wrecks.

All detections are mapped to a single class: **0 = Wreck**.

**Manual review with Label Studio is always recommended** to ensure annotation quality.

## 📊 Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `Wreck` | Shipwrecks, beached/rusted ships, stranded vessels, sunken structures visible from satellite |

## 🏋️ Training on Google Colab

1. Run `python scripts/prepare_dataset.py` locally
2. Create the zip: `cd dataset && zip -r ../dataset.zip images/ labels/ && cd ..`
3. Upload `dataset.zip` and `notebooks/train_colab.ipynb` to Colab
4. Set Runtime → T4 GPU
5. Execute all cells
6. Download `best.pt` and place it in `models/`

## 📄 License

This project is for educational/research purposes. Satellite images are subject to their respective providers' Terms of Service (Google, Mapbox).
