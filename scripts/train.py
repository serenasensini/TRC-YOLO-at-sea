#!/usr/bin/env python3
"""
Step 5 — Training YOLOv11 per Ship & Wreck Detection
Fine-tuning di YOLO11 sul dataset preparato.

Usa il modello pre-trainato yolo11m.pt (medium) per un buon compromesso
tra velocità e accuratezza.
"""

import argparse
from pathlib import Path

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"

# Training hyperparameters (tuned for small & imbalanced satellite dataset)
DEFAULT_MODEL = "yolo11m.pt"
DEFAULT_EPOCHS = 150        # Più epoche per dataset piccolo
DEFAULT_BATCH = 8           # Batch ridotto per dataset piccolo (~300 img)
DEFAULT_IMGSZ = 640
DEFAULT_PATIENCE = 30       # Più pazienza: convergenza lenta con pochi dati
DEFAULT_DEVICE = "0"        # GPU 0. Usa "cpu" se non hai GPU
# ----------------------------


def train(args):
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚢  YOLO at Sea — Training YOLOv11")
    print("=" * 60)
    print(f"   Model      : {args.model}")
    print(f"   Dataset    : {args.data}")
    print(f"   Epochs     : {args.epochs}")
    print(f"   Batch size : {args.batch}")
    print(f"   Image size : {args.imgsz}")
    print(f"   Device     : {args.device}")
    print(f"   Patience   : {args.patience}")
    print("=" * 60)

    # Load pre-trained model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(OUTPUT_DIR),
        name="yolo_at_sea",
        exist_ok=True,
        # Augmentation parameters ottimizzati per satellite imagery (dataset piccolo)
        mosaic=1.0,
        mixup=0.15,           # Leggermente più alto per aumentare variabilità
        copy_paste=0.15,      # Più copy-paste per compensare pochi esempi
        degrees=45.0,         # Rotazione ampia (satellite = vista dall'alto, rotazione-invariante)
        translate=0.15,
        scale=0.5,
        flipud=0.5,           # Flip verticale (utile per satellite)
        fliplr=0.5,           # Flip orizzontale
        hsv_h=0.015,
        hsv_s=0.6,            # Più variazione saturazione per condizioni meteo
        hsv_v=0.4,            # Più variazione luminosità
        erasing=0.1,          # Random erasing per robustezza
        # Learning rate (più basso per dataset piccolo, evita overfitting)
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=5,      # Warmup più lungo per stabilizzare
        # Optimizer
        optimizer="AdamW",
        weight_decay=0.001,   # Più weight decay per regolarizzazione
        # Loss weights (boost classification loss per dataset sbilanciato)
        cls=1.5,              # Aumenta peso class loss
        # Save
        save=True,
        save_period=10,
        plots=True,
        # Regolarizzazione extra
        dropout=0.1,          # Dropout per ridurre overfitting
    )

    # Copy best weights
    best_pt = OUTPUT_DIR / "yolo_at_sea" / "weights" / "best.pt"
    if best_pt.exists():
        dest = MODELS_DIR / "best.pt"
        import shutil
        shutil.copy2(str(best_pt), str(dest))
        print(f"\n✅  Best weights saved to: {dest}")

    # Validation
    print("\n📊  Running validation...")
    model_best = YOLO(str(best_pt)) if best_pt.exists() else model
    val_results = model_best.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
    )

    print(f"\n📊  Validation Results:")
    print(f"   mAP50    : {val_results.box.map50:.4f}")
    print(f"   mAP50-95 : {val_results.box.map:.4f}")

    print(f"\n💡  Prossimo step: python scripts/stream_demo.py")
    print(f"   Training curves: {OUTPUT_DIR / 'yolo_at_sea'}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Ship & Wreck Detection")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Pre-trained model to fine-tune")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML), help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Image size")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device: '0' for GPU, 'cpu' for CPU")
    args = parser.parse_args()
    args.data = Path(args.data)

    train(args)


if __name__ == "__main__":
    main()

