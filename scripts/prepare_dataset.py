#!/usr/bin/env python3
"""
Step 4 — Preparazione dataset per YOLOv11
Unisce immagini (Kaggle + satellitari) con le label auto-generate,
esegue split train/val 80/20 e applica data augmentation base.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = PROJECT_ROOT / "dataset" / "kaggle_raw"
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
ENABLE_AUGMENTATION = True
BALANCE_CLASSES = True  # Oversample minority classes
# ----------------------------


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


def _get_dominant_class(label_path: Optional[Path]) -> Optional[int]:
    """Restituisce la classe dominante (più frequente) in un file label."""
    if label_path is None or not label_path.exists():
        return None
    lines = label_path.read_text().strip().split("\n")
    classes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
    if not classes:
        return None
    from collections import Counter
    return Counter(classes).most_common(1)[0][0]


def balance_pairs(pairs: list[tuple[Path, Optional[Path]]]) -> list[tuple[Path, Optional[Path]]]:
    """
    Oversampling delle classi minoritarie per bilanciare il dataset.
    Raggruppa per classe dominante e replica le classi sotto-rappresentate.
    """
    from collections import Counter, defaultdict

    class_groups: dict[Optional[int], list] = defaultdict(list)
    for pair in pairs:
        cls = _get_dominant_class(pair[1])
        class_groups[cls].append(pair)

    # Statistiche
    print("\n📊  Class distribution (before balancing):")
    for cls_id, group in sorted(class_groups.items(), key=lambda x: (x[0] is None, x[0])):
        label = {0: "ship", 1: "wreck", 2: "sea"}.get(cls_id, "background/empty")
        print(f"   Class {cls_id} ({label}): {len(group)} images")

    # Trova la dimensione della classe più grande (escludendo background)
    labeled_groups = {k: v for k, v in class_groups.items() if k is not None}
    if not labeled_groups:
        print("⚠️  No labeled images found, skipping balancing")
        return pairs

    max_count = max(len(v) for v in labeled_groups.values())

    balanced = list(class_groups.get(None, []))  # Background images as-is
    for cls_id, group in labeled_groups.items():
        if len(group) < max_count:
            # Oversample: ripeti ciclicamente fino a raggiungere max_count
            oversampled = group.copy()
            while len(oversampled) < max_count:
                oversampled.extend(random.sample(group, min(len(group), max_count - len(oversampled))))
            balanced.extend(oversampled)
            label = {0: "ship", 1: "wreck", 2: "sea"}.get(cls_id, str(cls_id))
            print(f"   ⬆️  Oversampled class {cls_id} ({label}): {len(group)} → {max_count}")
        else:
            balanced.extend(group)

    print(f"   Total after balancing: {len(balanced)} (was {len(pairs)})")
    return balanced


def collect_image_label_pairs() -> list[tuple[Path, Optional[Path]]]:
    """Raccoglie coppie (immagine, label). Label può essere None se mancante."""
    extensions = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
    pairs = []

    for source_dir in [KAGGLE_DIR, SATELLITE_DIR]:
        if not source_dir.exists():
            continue
        for img_path in sorted(source_dir.rglob("*")):
            if img_path.suffix.lower() not in extensions:
                continue
            # Cerca label corrispondente (Label Studio ha priorità)
            # Label Studio exports may be in a labels/ subdir with UUID prefixes
            label_path_studio = _find_label_studio_file(img_path.stem)
            label_path_auto = LABEL_DIR / f"{img_path.stem}.txt"
            if label_path_studio is not None:
                pairs.append((img_path, label_path_studio))
            elif label_path_auto.exists():
                pairs.append((img_path, label_path_auto))
            else:
                # Immagine senza label → immagine di background (utile per ridurre FP)
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

    # Balance classes via oversampling
    if BALANCE_CLASSES:
        pairs = balance_pairs(pairs)

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
            label_lines = [l.strip() for l in label_path.read_text().strip().split("\n") if l.strip()]
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
            label_lines = [l.strip() for l in label_path.read_text().strip().split("\n") if l.strip()]
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

