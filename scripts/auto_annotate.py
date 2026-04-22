#!/usr/bin/env python3
"""
Step 3 — Auto-annotazione per Wreck Detection
Genera bounding box in formato YOLO per le immagini satellitari.

Strategia:
  - Grounding DINO + SAM2: zero-shot detection con prompt testuali per wrecks
  - Fallback: YOLOv8 pre-trainato su COCO (rileva barche come proxy)
  - Output: file .txt in formato YOLO (class x_center y_center w h)

Classe unica: 0=wreck
"""

from pathlib import Path

import cv2
from tqdm import tqdm

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SATELLITE_DIR = PROJECT_ROOT / "dataset" / "satellite_raw_old"
OUTPUT_LABEL_DIR = PROJECT_ROOT / "dataset" / "labels_auto"
CONFIDENCE_THRESHOLD = 0.25
# ----------------------------

# Prompt di detection per Grounding DINO (solo wrecks)
WRECK_PROMPTS = ["shipwreck", "sunken ship", "wreck", "rusted ship", "beached ship",
                 "abandoned ship", "stranded vessel"]


def collect_images() -> list[Path]:
    """Raccoglie le immagini satellitari."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp")
    images = []
    for ext in extensions:
        images.extend(SATELLITE_DIR.rglob(ext))
    print(f"📸  Found {len(images)} satellite images")
    return sorted(images)


def bbox_to_yolo(bbox, img_w: int, img_h: int, class_id: int = 0) -> str:
    """Converte bbox [x1, y1, x2, y2] in formato YOLO normalizzato."""
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clip to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def try_autodistill_annotation(images: list[Path]) -> bool:
    """
    Metodo A: usa autodistill con Grounded SAM2 per annotazione zero-shot.
    Restituisce True se riuscito, False se il pacchetto non è disponibile.
    """
    try:
        from autodistill_grounded_sam_2 import GroundedSAM2
        from autodistill.detection import CaptionOntology
        import supervision as sv
    except ImportError:
        print("⚠️  autodistill-grounded-sam-2 non disponibile, uso metodo fallback")
        return False

    print("🔍  Usando Grounding DINO + SAM2 (autodistill) per auto-annotazione wrecks...")

    # Define ontology: all prompts map to wreck (single class)
    ontology = CaptionOntology({
        "shipwreck": "wreck",
        "sunken ship": "wreck",
        "wreck": "wreck",
        "rusted ship": "wreck",
        "beached ship": "wreck",
        "abandoned ship": "wreck",
        "stranded vessel": "wreck",
    })

    base_model = GroundedSAM2(ontology=ontology)
    annotated = 0

    for img_path in tqdm(images, desc="Auto-annotating (GSAM2)"):
        try:
            result = base_model.predict(str(img_path))

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            label_lines = []
            if result.xyxy is not None and len(result.xyxy) > 0:
                for i, bbox in enumerate(result.xyxy):
                    conf = result.confidence[i] if result.confidence is not None else 1.0
                    if conf < CONFIDENCE_THRESHOLD:
                        continue
                    # All detections are class 0 (wreck)
                    yolo_line = bbox_to_yolo(bbox, img_w, img_h, class_id=0)
                    label_lines.append(yolo_line)

            # Save label file
            label_path = OUTPUT_LABEL_DIR / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            if label_lines:
                annotated += 1

        except Exception as e:
            print(f"   ⚠️  Error on {img_path.name}: {e}")
            continue

    print(f"✅  Grounded SAM2: annotated {annotated}/{len(images)} images with wreck detections")
    return True


def try_ultralytics_annotation(images: list[Path]) -> bool:
    """
    Metodo B (fallback): usa un modello YOLO pre-trainato su COCO
    per rilevare barche (classe 'boat' in COCO = id 8) come proxy per wrecks.
    Richiede review manuale successiva.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics non disponibile!")
        return False

    print("🔍  Usando YOLOv8n pre-trainato (COCO) come fallback...")
    print("    ℹ️  Rileverà 'boat' da COCO e le annoterà come 'wreck' (classe 0)")
    print("    ℹ️  Review manuale fortemente consigliata!")

    model = YOLO("yolov8n.pt")

    # In COCO, 'boat' è classe 8
    COCO_BOAT_ID = 8
    annotated = 0

    for img_path in tqdm(images, desc="Auto-annotating (YOLOv8)"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            results = model.predict(
                str(img_path),
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            label_lines = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != COCO_BOAT_ID:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    yolo_line = bbox_to_yolo(xyxy, img_w, img_h, class_id=0)
                    label_lines.append(yolo_line)

            label_path = OUTPUT_LABEL_DIR / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            if label_lines:
                annotated += 1

        except Exception as e:
            print(f"   ⚠️  Error on {img_path.name}: {e}")
            continue

    print(f"✅  YOLOv8 fallback: annotated {annotated}/{len(images)} images")
    print("⚠️  NOTA: tutte le detection sono 'wreck' (0) — review manuale consigliata.")
    return True


def generate_annotation_report():
    """Genera un report sulle annotazioni prodotte."""
    label_files = list(OUTPUT_LABEL_DIR.glob("*.txt"))
    total = len(label_files)
    non_empty = 0
    total_boxes = 0

    for lf in label_files:
        lines = lf.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]
        if lines:
            non_empty += 1
            total_boxes += len(lines)

    print(f"\n📊  Annotation Report:")
    print(f"   Total label files  : {total}")
    print(f"   With detections    : {non_empty}")
    print(f"   Empty (background) : {total - non_empty}")
    print(f"   Total bounding boxes: {total_boxes}")
    print(f"   Class 0 (wreck)    : {total_boxes}")
    print(f"   Label directory    : {OUTPUT_LABEL_DIR}")


def main():
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    images = collect_images()

    if not images:
        print("❌  Nessuna immagine trovata! Esegui prima:")
        print("   python scripts/download_satellite.py --provider mapbox --api-key YOUR_TOKEN")
        return

    # Annota immagini satellitari (wreck detection)
    print(f"\n🛰️   Annotating {len(images)} satellite images for wreck detection...")
    success = try_autodistill_annotation(images)
    if not success:
        success = try_ultralytics_annotation(images)

    generate_annotation_report()
    print("\n💡  PROSSIMO STEP: rivedi le annotazioni con Label Studio o CVAT")
    print("    Poi esegui: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()

