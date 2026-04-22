#!/usr/bin/env python3
"""
Step 5 — Training YOLOv11 per Wreck Detection
Fine-tuning di YOLO11 sul dataset preparato.
Classe unica: 0=wreck

Usa il modello pre-trainato yolo11n.pt (nano) — più adatto a dataset piccoli.
"""

import argparse
from pathlib import Path

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "configs" / "dataset.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"

# Training hyperparameters (tuned for small & imbalanced satellite dataset)
DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_EPOCHS = 200        # Più epoche: convergenza lenta con dataset piccolo + freeze
DEFAULT_BATCH = 8           # Batch ridotto per dataset piccolo
DEFAULT_IMGSZ = 640
DEFAULT_PATIENCE = 40       # Più pazienza: con freeze+lr basso la convergenza è graduale
DEFAULT_DEVICE = "0"        # GPU 0. Usa "cpu" se non hai GPU
# ----------------------------


def train(args):
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚢  YOLO at Sea — Training YOLOv11 (Wreck Detection)")
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
        # SINGLE CLASS — fondamentale per detection a classe unica
        single_cls=True,
        # Freeze backbone (primi 10 layer) per evitare overfitting su dataset piccolo
        freeze=10,
        # Augmentation — aggressive per compensare dataset piccolo
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        degrees=20.0,         # Rotazione più ampia (wreck possono avere qualsiasi angolo)
        translate=0.2,
        scale=0.5,            # Scale più ampio per variabilità dimensioni
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.3,
        erasing=0.2,
        # Learning rate — più basso per fine-tuning stabile
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=10,     # Warmup più lungo per stabilizzare con freeze
        # Optimizer
        optimizer="AdamW",
        weight_decay=0.0005,
        # Save
        save=True,
        save_period=10,
        plots=True,
        # Regolarizzazione
        dropout=0.15,
        # Close mosaic più tardi per sfruttare augmentation più a lungo
        close_mosaic=20,
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
    parser = argparse.ArgumentParser(description="Train YOLOv11 for Wreck Detection")
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

