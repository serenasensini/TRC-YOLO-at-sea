#!/usr/bin/env python3
"""
Step 4 — Preparazione dataset per YOLOv11
Unisce immagini satellitari con le label (wreck detection, classe unica),
esegue split train/val 80/20 e applica data augmentation base.

Classe unica: 0=wreck
"""

import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SATELLITE_DIR = PROJECT_ROOT / "dataset" / "satellite_raw"
LABEL_DIR = PROJECT_ROOT / "dataset" / "labels_auto"
LABEL_STUDIO_DIR = PROJECT_ROOT / "dataset" / "labels_studio"  # Label Studio YOLO export

IMAGES_TRAIN = PROJECT_ROOT / "dataset" / "images" / "train"
IMAGES_VAL = PROJECT_ROOT / "dataset" / "images" / "val"
LABELS_TRAIN = PROJECT_ROOT / "dataset" / "labels" / "train"
LABELS_VAL = PROJECT_ROOT / "dataset" / "labels" / "val"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
TARGET_SIZE = 640  # Resize images to 640x640 for YOLO
ENABLE_AUGMENTATION = False  # Disabilitato: YOLO ha già augmentation online, evita duplicazione
# Only wreck class is kept (class 0 in both Label Studio and auto labels)
# ----------------------------


def _detect_label_source(label_path: Path) -> str:
    """Detect which labeling scheme a file uses based on its location."""
    path_str = str(label_path)
    if "labels_studio" in path_str:
        return "studio"  # classes.txt: Wreck=0 (polygon format)
    return "auto"  # Current auto_annotate: wreck=0 (bbox format)


def _polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert polygon coordinates [x1,y1,x2,y2,...] to YOLO bbox [xc, yc, w, h]."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return xc, yc, w, h


def _filter_labels(label_lines: list[str], source: str = "auto") -> list[str]:
    """
    Keeps only wreck annotations and remaps to class 0.
    Handles both bbox format (5 values) and polygon format (>5 values).

    Source mappings:
      - "studio": Label Studio export → Wreck=0 (polygon format, converted to bbox)
      - "auto": auto_annotate.py → Wreck=0 (bbox format)
    """
    wreck_ids = {0}  # Both sources use class 0 for wreck

    filtered = []
    for l in label_lines:
        if not l.strip():
            continue
        parts = l.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        if cls_id not in wreck_ids:
            continue

        coords = [float(p) for p in parts[1:]]
        if len(coords) == 4:
            # Already bbox format: xc yc w h
            filtered.append(f"0 {' '.join(parts[1:])}")
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            # Polygon format: x1 y1 x2 y2 ... → convert to bbox
            xc, yc, w, h = _polygon_to_bbox(coords)
            filtered.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return filtered


def _find_label_studio_file(stem: str) -> Optional[Path]:
    """
    Cerca un file label di Label Studio che corrisponda allo stem dell'immagine.
    Label Studio esporta file con prefisso UUID, es: 'abc123-nome_originale.txt',
    e può metterli in una sottodirectory 'labels/'.
    """
    # Cerca nelle possibili directory
    for search_dir in [LABEL_STUDIO_DIR, LABEL_STUDIO_DIR / "labels"]:
        if not search_dir.exists():
            continue
        # Match esatto
        exact = search_dir / f"{stem}.txt"
        if exact.exists():
            return exact
        # Match con prefisso UUID (pattern: UUID-stem.txt)
        for f in search_dir.glob(f"*-{stem}.txt"):
            return f
    return None



def collect_image_label_pairs() -> list[tuple[Path, Optional[Path]]]:
    """Raccoglie coppie (immagine, label). Label può essere None se mancante."""
    extensions = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
    pairs = []

    if not SATELLITE_DIR.exists():
        return pairs
    for img_path in sorted(SATELLITE_DIR.rglob("*")):
        if img_path.suffix.lower() not in extensions:
            continue
        # Cerca label corrispondente (Label Studio ha priorità)
        label_path_studio = _find_label_studio_file(img_path.stem)
        label_path_auto = LABEL_DIR / f"{img_path.stem}.txt"
        if label_path_studio is not None:
            pairs.append((img_path, label_path_studio))
        elif label_path_auto.exists():
            pairs.append((img_path, label_path_auto))
        else:
            # Immagine senza label → background (no wreck)
            pairs.append((img_path, None))

    print(f"📁  Found {len(pairs)} image-label pairs")
    with_labels = sum(1 for _, l in pairs if l is not None)
    print(f"   With labels    : {with_labels}")
    print(f"   Background only: {len(pairs) - with_labels}")
    return pairs


def resize_image(img: np.ndarray, target_size: int) -> np.ndarray:
    """Ridimensiona l'immagine a target_size x target_size mantenendo l'aspect ratio con padding."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding per ottenere un quadrato
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def adjust_labels_for_resize(label_lines: list[str], orig_w: int, orig_h: int,
                              target_size: int) -> list[str]:
    """Ricalcola le coordinate YOLO dopo il resize con padding."""
    scale = target_size / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (target_size - new_w) / 2
    y_offset = (target_size - new_h) / 2

    adjusted = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = parts[0]
        xc, yc, w, h = [float(p) for p in parts[1:]]

        # Da normalizzato originale → pixel originali
        xc_px = xc * orig_w
        yc_px = yc * orig_h
        w_px = w * orig_w
        h_px = h * orig_h

        # Applica scale + offset
        xc_new = (xc_px * scale + x_offset) / target_size
        yc_new = (yc_px * scale + y_offset) / target_size
        w_new = (w_px * scale) / target_size
        h_new = (h_px * scale) / target_size

        adjusted.append(f"{cls_id} {xc_new:.6f} {yc_new:.6f} {w_new:.6f} {h_new:.6f}")

    return adjusted


def augment_image_and_labels(img: np.ndarray, label_lines: list[str],
                              stem: str) -> list[tuple[np.ndarray, list[str], str]]:
    """
    Genera varianti augmentate dell'immagine.
    Restituisce lista di (immagine, label_lines, nuovo_stem).
    """
    augmented = []

    # 1. Horizontal flip
    flipped_h = cv2.flip(img, 1)
    flipped_labels_h = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = parts[0]
        xc, yc, w, h = [float(p) for p in parts[1:]]
        xc = 1.0 - xc  # Flip orizzontale
        flipped_labels_h.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    augmented.append((flipped_h, flipped_labels_h, f"{stem}_hflip"))

    # 2. Brightness jitter
    brightness_delta = random.randint(-30, 30)
    bright = np.clip(img.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)
    augmented.append((bright, label_lines.copy(), f"{stem}_bright"))

    # 3. 90° rotation
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot_labels = []
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = parts[0]
        xc, yc, w, h = [float(p) for p in parts[1:]]
        # 90° CW: new_xc = 1 - yc, new_yc = xc, new_w = h, new_h = w
        new_xc = 1.0 - yc
        new_yc = xc
        rot_labels.append(f"{cls_id} {new_xc:.6f} {new_yc:.6f} {h:.6f} {w:.6f}")
    augmented.append((rotated, rot_labels, f"{stem}_rot90"))

    return augmented


def save_pair(img: np.ndarray, label_lines: list[str],
              stem: str, img_dir: Path, lbl_dir: Path):
    """Salva coppia immagine + label."""
    img_path = img_dir / f"{stem}.png"
    cv2.imwrite(str(img_path), img)

    lbl_path = lbl_dir / f"{stem}.txt"
    with open(lbl_path, "w") as f:
        f.write("\n".join(label_lines))


def main():
    random.seed(RANDOM_SEED)

    # Pulisci directory output
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    pairs = collect_image_label_pairs()
    if not pairs:
        print("❌  Nessuna immagine trovata! Esegui prima gli step precedenti.")
        return


    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"\n📊  Split: {len(train_pairs)} train / {len(val_pairs)} val")

    # Process train
    train_count = 0
    print("\n🔧  Processing training set...")
    for img_path, label_path in tqdm(train_pairs, desc="Train"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Load labels
        if label_path:
            source = _detect_label_source(label_path)
            label_lines = [l.strip() for l in label_path.read_text().strip().split("\n") if l.strip()]
            label_lines = _filter_labels(label_lines, source)
        else:
            label_lines = []

        # Resize
        img_resized = resize_image(img, TARGET_SIZE)
        if label_lines:
            label_lines = adjust_labels_for_resize(label_lines, orig_w, orig_h, TARGET_SIZE)

        stem = img_path.stem
        save_pair(img_resized, label_lines, stem, IMAGES_TRAIN, LABELS_TRAIN)
        train_count += 1

        # Augmentation (solo per immagini con label)
        if ENABLE_AUGMENTATION and label_lines:
            aug_variants = augment_image_and_labels(img_resized, label_lines, stem)
            for aug_img, aug_labels, aug_stem in aug_variants:
                save_pair(aug_img, aug_labels, aug_stem, IMAGES_TRAIN, LABELS_TRAIN)
                train_count += 1

    # Process val (no augmentation)
    val_count = 0
    print("\n🔧  Processing validation set...")
    for img_path, label_path in tqdm(val_pairs, desc="Val"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        if label_path:
            source = _detect_label_source(label_path)
            label_lines = [l.strip() for l in label_path.read_text().strip().split("\n") if l.strip()]
            label_lines = _filter_labels(label_lines, source)
        else:
            label_lines = []

        img_resized = resize_image(img, TARGET_SIZE)
        if label_lines:
            label_lines = adjust_labels_for_resize(label_lines, orig_w, orig_h, TARGET_SIZE)

        save_pair(img_resized, label_lines, img_path.stem, IMAGES_VAL, LABELS_VAL)
        val_count += 1

    print(f"\n✅  Dataset pronto!")
    print(f"   Train: {train_count} images (con augmentation)")
    print(f"   Val  : {val_count} images")
    print(f"   Output: {PROJECT_ROOT / 'dataset'}")
    print(f"\n💡  Prossimo step: python scripts/train.py")


if __name__ == "__main__":
    main()

