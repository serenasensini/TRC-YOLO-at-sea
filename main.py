#!/usr/bin/env python3
"""
🚢 YOLO at Sea — Real-Time AI for Wreck Detection
===================================================

Pipeline orchestrator: esegue tutti gli step in sequenza.
Ogni step può essere eseguito anche singolarmente.

Usage:
    python main.py --step all           # Esegue tutto
    python main.py --step download      # Solo download satellite
    python main.py --step annotate      # Solo auto-annotazione
    python main.py --step prepare       # Solo preparazione dataset
    python main.py --step train         # Solo training
    python main.py --step demo          # Solo demo streaming
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = {
    "download_satellite": {
        "script": "scripts/download_satellite.py",
        "description": "🛰️  Step 1: Download satellite images (Google Maps / Mapbox)",
        "requires_manual": "Google Static Maps API key or Mapbox access token",
    },
    "annotate": {
        "script": "scripts/auto_annotate.py",
        "description": "🔍 Step 2: Auto-annotate wreck images (Grounding DINO + SAM2)",
        "requires_manual": None,
    },
    "prepare": {
        "script": "scripts/prepare_dataset.py",
        "description": "📦 Step 3: Prepare YOLO dataset (split + resize)",
        "requires_manual": None,
    },
    "train": {
        "script": "scripts/train.py",
        "description": "🏋️  Step 4: Train YOLOv11 (wreck detection)",
        "requires_manual": None,
    },
    "demo": {
        "script": "scripts/stream_demo.py",
        "description": "🎬 Step 5: Real-time streaming demo",
        "requires_manual": None,
    },
}

# Predefined step groups
STEP_GROUPS = {
    "all": list(STEPS.keys()),
    "download": ["download_satellite"],
    "pipeline": ["annotate", "prepare", "train"],
    "full": list(STEPS.keys()),
}


def run_step(step_name: str, extra_args: list[str] = None):
    """Esegue un singolo step."""
    step = STEPS[step_name]
    script = PROJECT_ROOT / step["script"]

    print("\n" + "=" * 60)
    print(step["description"])
    print("=" * 60)

    if step.get("requires_manual"):
        print(f"⚠️  Requisito manuale: {step['requires_manual']}")

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n❌  Step '{step_name}' fallito con codice {result.returncode}")
        return False
    return True


def show_pipeline_status():
    """Mostra lo stato della pipeline."""
    print("\n🚢 YOLO at Sea — Pipeline Status")
    print("=" * 50)

    checks = {
        "Satellite images": (PROJECT_ROOT / "dataset" / "satellite_raw").exists()
                            and any((PROJECT_ROOT / "dataset" / "satellite_raw").glob("*.png")),
        "Auto-annotations": (PROJECT_ROOT / "dataset" / "labels_auto").exists()
                            and any((PROJECT_ROOT / "dataset" / "labels_auto").glob("*.txt")),
        "Train dataset": (PROJECT_ROOT / "dataset" / "images" / "train").exists()
                         and any((PROJECT_ROOT / "dataset" / "images" / "train").glob("*.*")),
        "Val dataset": (PROJECT_ROOT / "dataset" / "images" / "val").exists()
                       and any((PROJECT_ROOT / "dataset" / "images" / "val").glob("*.*")),
        "Trained model": (PROJECT_ROOT / "models" / "best.pt").exists(),
    }

    for name, ok in checks.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="🚢 YOLO at Sea — Wreck Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step groups:
  all        : Esegue tutti gli step (1-5)
  download   : Solo download satellite
  pipeline   : Solo processing (annotate + prepare + train)
  full       : Tutto

Single steps:
  download_satellite, annotate, prepare, train, demo

Examples:
  python main.py --step all
  python main.py --step demo
  python main.py --step train --extra "--epochs 50 --device cpu"
  python main.py --status
        """,
    )
    parser.add_argument("--step", type=str, default="status",
                        help="Step or group to execute")
    parser.add_argument("--extra", type=str, default="",
                        help="Extra arguments to pass to the step script")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status")
    args = parser.parse_args()

    print()
    print("  🚢🌊  YOLO at Sea  🌊🚢")
    print("  Real-Time AI for Wreck Detection")
    print()

    if args.status or args.step == "status":
        show_pipeline_status()
        return

    extra_args = args.extra.split() if args.extra else []

    # Resolve step group or single step
    if args.step in STEP_GROUPS:
        steps_to_run = STEP_GROUPS[args.step]
    elif args.step in STEPS:
        steps_to_run = [args.step]
    else:
        print(f"❌  Step sconosciuto: '{args.step}'")
        print(f"   Steps disponibili: {', '.join(list(STEPS.keys()) + list(STEP_GROUPS.keys()))}")
        return

    print(f"📋  Steps da eseguire: {', '.join(steps_to_run)}")

    for step_name in steps_to_run:
        success = run_step(step_name, extra_args if step_name == steps_to_run[-1] else None)
        if not success:
            print(f"\n⛔  Pipeline interrotta allo step '{step_name}'")
            break

    show_pipeline_status()


if __name__ == "__main__":
    main()
