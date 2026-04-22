#!/usr/bin/env python3
"""
Step 2 — Download Satellite Images (Google Static Maps or Mapbox)
Scarica immagini satellitari ad alta risoluzione per ogni coordinata
di relitto nel file CSV.

Provider supportati:
  1. google  — Google Static Maps API (default)
     → Ottieni la key da https://console.cloud.google.com/apis/credentials
     → Abilita "Maps Static API" nel progetto GCP
  2. mapbox  — Mapbox Static Images API
     → Ottieni il token da https://account.mapbox.com/access-tokens/

Uso:
  # Google (default)
  python scripts/download_satellite.py --api-key YOUR_KEY

  # Mapbox (se Google non supporta satellite nella tua regione)
  python scripts/download_satellite.py --provider mapbox --api-key YOUR_MAPBOX_TOKEN

  # Oppure usa variabili d'ambiente:
  export GOOGLE_MAPS_API_KEY=...   # per google
  export MAPBOX_ACCESS_TOKEN=...   # per mapbox
"""

import os
import csv
import time
import argparse
import hashlib
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COORDS_CSV = PROJECT_ROOT / "configs" / "wreck_coordinates.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "satellite_raw_old"

# Common settings
IMAGE_SIZE = "640x640"
ZOOM_LEVELS = [17, 18, 19]  # 17=~1.2m/px, 18=~0.6m/px, 19=~0.3m/px
REQUEST_DELAY = 0.25         # seconds between requests (rate limiting)

# Google Static Maps API settings
GOOGLE_BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Mapbox Static Images API settings
MAPBOX_BASE_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"
# ----------------------------


def load_coordinates(csv_path: Path) -> list[dict]:
    """Carica le coordinate dal CSV."""
    coords = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append({
                "name": row["name"].strip(),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "desc": row.get("description", ""),
            })
    print(f"📍  Loaded {len(coords)} coordinates from {csv_path}")
    return coords


def download_image_google(lat: float, lon: float, zoom: int, api_key: str) -> Image.Image | None:
    """Scarica una singola immagine satellitare da Google Static Maps."""
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": IMAGE_SIZE,
        "maptype": "satellite",
        "key": api_key,
        "style": "feature:all|element:labels|visibility:off",
    }

    try:
        resp = requests.get(GOOGLE_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        if img.size[0] < 100:
            print(f"   ⚠️  Immagine troppo piccola, possibile errore API")
            return None
        return img

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Errore download: {e}")
        return None


def download_image_mapbox(lat: float, lon: float, zoom: int, api_key: str) -> Image.Image | None:
    """Scarica una singola immagine satellitare da Mapbox Static Images API."""
    # Mapbox format: /lon,lat,zoom/widthxheight@2x
    w, h = IMAGE_SIZE.split("x")
    url = f"{MAPBOX_BASE_URL}/{lon},{lat},{zoom}/{w}x{h}@2x"
    params = {"access_token": api_key}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        if img.size[0] < 100:
            print(f"   ⚠️  Immagine troppo piccola, possibile errore API")
            return None
        return img

    except requests.exceptions.RequestException as e:
        print(f"   ❌ Errore download: {e}")
        return None


def download_all(api_key: str, provider: str = "google"):
    """Scarica tutte le immagini satellitari."""
    download_fn = download_image_google if provider == "google" else download_image_mapbox

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coords = load_coordinates(COORDS_CSV)

    total = len(coords) * len(ZOOM_LEVELS)
    downloaded = 0
    skipped = 0

    print(f"🛰️   Downloading {total} images ({len(coords)} locations × {len(ZOOM_LEVELS)} zoom levels)...\n")

    with tqdm(total=total, desc="Downloading") as pbar:
        for coord in coords:
            for zoom in ZOOM_LEVELS:
                # Filename: name_z{zoom}_{hash}.png
                uid = hashlib.md5(f"{coord['lat']}_{coord['lon']}_{zoom}".encode()).hexdigest()[:8]
                filename = f"{coord['name']}_z{zoom}_{uid}.png"
                filepath = OUTPUT_DIR / filename

                if filepath.exists():
                    skipped += 1
                    pbar.update(1)
                    continue

                img = download_fn(coord["lat"], coord["lon"], zoom, api_key)

                if img is not None:
                    img.save(str(filepath), "PNG")
                    downloaded += 1
                else:
                    print(f"   ⚠️  Skipped: {coord['name']} zoom={zoom}")

                time.sleep(REQUEST_DELAY)
                pbar.update(1)

    print(f"\n✅  Download completato!")
    print(f"   Downloaded : {downloaded}")
    print(f"   Skipped    : {skipped} (già presenti)")
    print(f"   Output dir : {OUTPUT_DIR}")


def main():
    global COORDS_CSV

    parser = argparse.ArgumentParser(description="Download satellite images for wreck detection")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["google", "mapbox"],
        default="google",
        help="Satellite imagery provider: 'google' or 'mapbox' (default: google)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (or set GOOGLE_MAPS_API_KEY / MAPBOX_ACCESS_TOKEN env var)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(COORDS_CSV),
        help="Path to CSV with coordinates",
    )
    args = parser.parse_args()

    # Resolve API key from argument or environment variable
    api_key = args.api_key
    if not api_key:
        if args.provider == "google":
            api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        else:
            api_key = os.environ.get("MAPBOX_ACCESS_TOKEN", "")

    if not api_key:
        print("❌  API key mancante!")
        if args.provider == "google":
            print("   Usa: python scripts/download_satellite.py --api-key YOUR_KEY")
            print("   Oppure: export GOOGLE_MAPS_API_KEY=YOUR_KEY")
            print()
            print("   Per ottenere la key:")
            print("   1. Vai su https://console.cloud.google.com/apis/credentials")
            print("   2. Crea un progetto (o usane uno esistente)")
            print("   3. Abilita 'Maps Static API'")
            print("   4. Crea una API key")
        else:
            print("   Usa: python scripts/download_satellite.py --provider mapbox --api-key YOUR_TOKEN")
            print("   Oppure: export MAPBOX_ACCESS_TOKEN=YOUR_TOKEN")
            print()
            print("   Per ottenere il token:")
            print("   1. Vai su https://account.mapbox.com/access-tokens/")
            print("   2. Crea un token (il free tier include 50k richieste/mese)")
        return

    COORDS_CSV = Path(args.csv)

    print(f"🛰️   Provider: {args.provider}")
    download_all(api_key, provider=args.provider)


if __name__ == "__main__":
    main()

