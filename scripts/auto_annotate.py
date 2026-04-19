#!/usr/bin/env python3
"""
Step 3 — Auto-annotazione con strategie multiple
Genera bounding box in formato YOLO per tutte le immagini del dataset.

Strategia per immagini Kaggle (ships-in-satellite-imagery):
  - Le immagini 80×80 sono già classificate ship/no-ship dal filename.
  - Ship: bounding box centrato che copre ~70% dell'immagine (classe 0).
  - No-ship: label vuoto (immagine di background).

Strategia per immagini satellitari (Google/Mapbox):
  - Grounding DINO + SAM2: zero-shot object detection con prompt testuali
  - Fallback: YOLOv8 pre-trainato su COCO (rileva barche)
  - Output: file .txt in formato YOLO (class x_center y_center w h)


Classi: 0=ship, 1=wreck
"""

import os
from pathlib import Path

import cv2
from tqdm import tqdm

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = PROJECT_ROOT / "dataset" / "kaggle_raw"
SATELLITE_DIR = PROJECT_ROOT / "dataset" / "satellite_raw"
OUTPUT_LABEL_DIR = PROJECT_ROOT / "dataset" / "labels_auto"
CONFIDENCE_THRESHOLD = 0.25
# ----------------------------

# Prompt di detection per Grounding DINO
SHIP_PROMPTS = ["ship", "vessel", "boat", "cargo ship", "tanker"]
WRECK_PROMPTS = ["shipwreck", "sunken ship", "wreck", "rusted ship", "beached ship"]


def collect_images() -> tuple[list[Path], list[Path]]:
    """Raccoglie le immagini, separate per fonte (Kaggle vs satellite)."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp")
    kaggle_images = []
    satellite_images = []

    for ext in extensions:
        kaggle_images.extend(KAGGLE_DIR.rglob(ext))
        satellite_images.extend(SATELLITE_DIR.rglob(ext))

    print(f"📸  Found {len(kaggle_images)} Kaggle images + {len(satellite_images)} satellite images")
    return sorted(kaggle_images), sorted(satellite_images)


def bbox_to_yolo(bbox, img_w: int, img_h: int, class_id: int) -> str:
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


def annotate_kaggle_images(images: list[Path]) -> int:
    """
    Annota le immagini Kaggle ships-in-satellite-imagery.
    Le immagini sono 80×80 e il filename inizia con il label:
      - 1__... → ship (classe 0): bbox centrato ~70% dell'immagine
      - 0__... → no-ship: label vuoto (background)
    Se sono organizzate in sottocartelle ship/ e no_ship/, usa quelle.
    """
    print("🏷️   Annotating Kaggle ship/no-ship images...")
    annotated = 0

    for img_path in tqdm(images, desc="Kaggle labels"):
        # Determina se è ship o no-ship dal filename o dalla cartella
        is_ship = False
        if img_path.parent.name == "ship":
            is_ship = True
        elif img_path.parent.name == "no_ship":
            is_ship = False
        elif img_path.name.startswith("1__"):
            is_ship = True
        elif img_path.name.startswith("0__"):
            is_ship = False
        else:
            # Filename non riconosciuto, skip
            continue

        label_path = OUTPUT_LABEL_DIR / f"{img_path.stem}.txt"

        if is_ship:
            # Bounding box centrato che copre ~70% dell'immagine
            # YOLO format: class x_center y_center width height (normalizzati)
            label_lines = ["0 0.500000 0.500000 0.700000 0.700000"]
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
            annotated += 1
        else:
            # No-ship → file label vuoto (immagine di background)
            with open(label_path, "w") as f:
                f.write("")

    print(f"✅  Kaggle: {annotated} ship images annotated, rest as background")
    return annotated


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

    print("🔍  Usando Grounding DINO + SAM2 (autodistill) per auto-annotazione...")

    # Define ontology: map text prompts to class names
    ontology = CaptionOntology({
        "ship": "ship",
        "vessel": "ship",
        "boat": "ship",
        "cargo ship": "ship",
        "shipwreck": "wreck",
        "sunken ship": "wreck",
        "wreck": "wreck",
        "rusted ship": "wreck",
        "beached ship": "wreck",
    })

    base_model = GroundedSAM2(ontology=ontology)

    class_name_to_id = {"ship": 0, "wreck": 1}
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

                    class_name_idx = result.class_id[i] if result.class_id is not None else 0
                    # Map supervision class index to our class names
                    class_names = list(class_name_to_id.keys())
                    if class_name_idx < len(class_names):
                        class_id = class_name_to_id[class_names[class_name_idx]]
                    else:
                        class_id = 0

                    yolo_line = bbox_to_yolo(bbox, img_w, img_h, class_id)
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

    print(f"✅  Grounded SAM2: annotated {annotated}/{len(images)} images with detections")
    return True


def try_ultralytics_annotation(images: list[Path]) -> bool:
    """
    Metodo B (fallback): usa un modello YOLO pre-trainato su COCO
    per rilevare barche/navi (classe 'boat' in COCO = id 8).
    Non distingue ship vs wreck, ma è un buon punto di partenza.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌  ultralytics non disponibile!")
        return False

    print("🔍  Usando YOLOv8n pre-trainato (COCO) come fallback...")
    print("    ℹ️  Rileverà 'boat' da COCO e le annoterà come 'ship' (classe 0)")
    print("    ℹ️  Per la classe 'wreck', sarà necessaria annotazione manuale")

    model = YOLO("yolov8n.pt")  # Download automatico se non presente

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
                    conf = float(box.conf[0])

                    # Prendiamo solo 'boat' da COCO
                    if cls_id != COCO_BOAT_ID:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    # Tutte le barche come classe 0 (ship) - wreck richiede review manuale
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

    print(f"✅  YOLOv8 fallback: annotated {annotated}/{len(images)} images with boat detections")
    print("⚠️  NOTA: tutte le detection sono classe 'ship' (0).")
    print("   Per avere la classe 'wreck' (1), dovrai rivedere manualmente le label")
    print("   con uno strumento come Label Studio o CVAT.")
    return True


def generate_annotation_report():
    """Genera un report sulle annotazioni prodotte."""
    label_files = list(OUTPUT_LABEL_DIR.glob("*.txt"))
    total = len(label_files)
    non_empty = 0
    total_boxes = 0
    class_counts = {0: 0, 1: 0, 2: 0}

    for lf in label_files:
        lines = lf.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]
        if lines:
            non_empty += 1
            total_boxes += len(lines)
            for line in lines:
                parts = line.split()
                if parts:
                    cls = int(parts[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n📊  Annotation Report:")
    print(f"   Total label files  : {total}")
    print(f"   With detections    : {non_empty}")
    print(f"   Empty (background) : {total - non_empty}")
    print(f"   Total bounding boxes: {total_boxes}")
    print(f"   Class 0 (ship)     : {class_counts.get(0, 0)}")
    print(f"   Class 1 (wreck)    : {class_counts.get(1, 0)}")
    print(f"   Class 2 (sea)      : {class_counts.get(2, 0)}")
    print(f"   Label directory    : {OUTPUT_LABEL_DIR}")


def main():
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_images, satellite_images = collect_images()

    if not kaggle_images and not satellite_images:
        print("❌  Nessuna immagine trovata! Esegui prima:")
        print("   python scripts/download_kaggle.py")
        print("   python scripts/download_satellite.py --provider mapbox --api-key YOUR_TOKEN")
        return

    # 1. Annota immagini Kaggle (ship/no-ship già classificate)
    if kaggle_images:
        annotate_kaggle_images(kaggle_images)

    # 2. Annota immagini satellitari (richiedono object detection)
    if satellite_images:
        print(f"\n🛰️   Annotating {len(satellite_images)} satellite images...")
        success = try_autodistill_annotation(satellite_images)
        if not success:
            success = try_ultralytics_annotation(satellite_images)

    generate_annotation_report()
    print("\n💡  PROSSIMO STEP: rivedi un campione di annotazioni con:")
    print("    - Label Studio: https://labelstud.io/")
    print("    - CVAT: https://www.cvat.ai/")
    print("    Poi esegui: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()

