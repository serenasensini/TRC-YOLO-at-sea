#!/usr/bin/env python3
"""
Step 1 — Download Kaggle Ships in Satellite Imagery Dataset
Scarica e estrae il dataset Kaggle 'rhammell/ships-in-satellite-imagery'
in dataset/kaggle_raw/.

Il dataset contiene ~4000 immagini satellitari 80×80 px da Planet Labs,
classificate come 'ship' (1) o 'no-ship' (0).
Il filename di ogni immagine ha formato: {label}__{scene_id}__{lon}_{lat}.png

Prerequisito: token Kaggle in ~/.kaggle/kaggle.json
  → Ottienilo da https://www.kaggle.com/settings → "Create New Token"
"""

import os
import json
import zipfile
import glob
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ---------- CONFIG ----------
DATASET_SLUG = "rhammell/ships-in-satellite-imagery"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "kaggle_raw"
# ----------------------------


def check_kaggle_credentials():
    """Verifica che il token Kaggle sia configurato."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌  Token Kaggle non trovato!")
        print("   1. Vai su https://www.kaggle.com/settings")
        print("   2. Clicca 'Create New Token' → scarica kaggle.json")
        print(f"   3. Copialo in {kaggle_json}")
        print(f"   4. chmod 600 {kaggle_json}")
        return False
    return True


def organize_images():
    """
    Organizza le immagini in sottocartelle ship/ e no_ship/
    basandosi sul prefisso del filename (1__ = ship, 0__ = no-ship).
    Se presente shipsnet.json, lo usa per generare le immagini PNG.
    """
    ship_dir = OUTPUT_DIR / "ship"
    no_ship_dir = OUTPUT_DIR / "no_ship"
    ship_dir.mkdir(parents=True, exist_ok=True)
    no_ship_dir.mkdir(parents=True, exist_ok=True)

    # Se esiste shipsnet.json, genera le immagini PNG
    json_file = None
    for jf in OUTPUT_DIR.rglob("shipsnet.json"):
        json_file = jf
        break

    if json_file and json_file.exists():
        print("📄  Trovato shipsnet.json, generando immagini PNG...")
        with open(json_file, "r") as f:
            data = json.load(f)

        labels = data["labels"]
        pixel_data = data["data"]
        scene_ids = data.get("scene_ids", ["unknown"] * len(labels))
        locations = data.get("locations", [[0, 0]] * len(labels))

        for i, (label, pixels) in enumerate(zip(labels, pixel_data)):
            # Ricostruisci immagine 80×80×3 (R, G, B canali in sequenza)
            arr = np.array(pixels, dtype=np.uint8).reshape(3, 80, 80)
            arr = arr.transpose(1, 2, 0)  # HWC
            img = Image.fromarray(arr)

            scene = scene_ids[i] if i < len(scene_ids) else "unknown"
            loc = locations[i] if i < len(locations) else [0, 0]
            fname = f"{label}__{scene}__{loc[0]}_{loc[1]}.png"

            dest = ship_dir if label == 1 else no_ship_dir
            img.save(str(dest / fname))

        print(f"   ✅ Generati {len(labels)} PNG da shipsnet.json")
        json_file.unlink()
        return

    # Altrimenti, organizza i PNG già estratti in base al filename
    png_files = list(OUTPUT_DIR.glob("*.png"))
    # Cerca anche in sottocartelle (es. shipsnet/shipsnet/)
    png_files.extend(OUTPUT_DIR.rglob("*.png"))
    png_files = list(set(png_files))  # Rimuovi duplicati

    moved = 0
    for img_path in png_files:
        if img_path.parent in (ship_dir, no_ship_dir):
            continue  # Già organizzato
        name = img_path.name
        if name.startswith("1__"):
            shutil.move(str(img_path), str(ship_dir / name))
            moved += 1
        elif name.startswith("0__"):
            shutil.move(str(img_path), str(no_ship_dir / name))
            moved += 1

    if moved:
        print(f"   ✅ Organizzate {moved} immagini in ship/ e no_ship/")

    # Rimuovi sottocartelle vuote rimaste
    for dirpath in sorted(OUTPUT_DIR.rglob("*"), reverse=True):
        if dirpath.is_dir() and dirpath not in (ship_dir, no_ship_dir):
            try:
                dirpath.rmdir()  # Solo se vuota
            except OSError:
                pass


def download_dataset():
    """Scarica il dataset da Kaggle."""
    if not check_kaggle_credentials():
        return

    # Import kaggle API only after credentials check
    from kaggle.api.kaggle_api_extended import KaggleApi

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📥  Downloading dataset: {DATASET_SLUG}")
    api = KaggleApi()
    api.authenticate()

    # Download
    api.dataset_download_files(
        DATASET_SLUG,
        path=str(OUTPUT_DIR),
        unzip=False,
    )

    # Find and extract zip
    zip_files = glob.glob(str(OUTPUT_DIR / "*.zip"))
    if zip_files:
        for zf in zip_files:
            print(f"📦  Extracting {zf}...")
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(str(OUTPUT_DIR))
            os.remove(zf)
            print(f"   ✅ Extracted & removed {zf}")

    # Organizza in ship/ e no_ship/
    organize_images()

    # Report contents
    ship_count = len(list((OUTPUT_DIR / "ship").rglob("*.png"))) if (OUTPUT_DIR / "ship").exists() else 0
    no_ship_count = len(list((OUTPUT_DIR / "no_ship").rglob("*.png"))) if (OUTPUT_DIR / "no_ship").exists() else 0

    print(f"\n📊  Dataset summary:")
    print(f"   Ship images    : {ship_count}")
    print(f"   No-ship images : {no_ship_count}")
    print(f"   Total          : {ship_count + no_ship_count}")
    print(f"   Output dir     : {OUTPUT_DIR}")


if __name__ == "__main__":
    download_dataset()

