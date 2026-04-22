#!/usr/bin/env python3
"""
Step 6 — Real-Time Streaming Demo
Simula un ambiente di sorveglianza marittima in tempo reale,
processando immagini satellitari una alla volta come se arrivassero
da un feed live.

Features:
  - Carica il modello addestrato (best.pt)
  - Scansiona immagini da una cartella (simula stream satellitare)
  - Visualizza detection con bounding box, classe e confidence
  - Mostra statistiche live: FPS, conteggio detection, heatmap
  - Opzionalmente salva un video di output
"""

import time
import argparse
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

# Check if GUI display is available
HEADLESS = False
try:
    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imshow("_test", test_img)
    cv2.destroyWindow("_test")
except cv2.error:
    HEADLESS = True
    print("⚠️  No GUI display available — running in headless mode (video-only output)")

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "satellite_raw_old"
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / "runs" / "demo_output.mp4"
CONFIDENCE_THRESHOLD = 0.25
DISPLAY_DELAY = 3.0  # Secondi tra un frame e l'altro (simula latenza satellite)
WINDOW_NAME = "🚢 YOLO at Sea — Real-Time Maritime Surveillance"
# ----------------------------

# Colori per le classi (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),    # wreck → rosso
}
CLASS_NAMES = {0: "Wreck"}


def create_info_panel(frame_idx: int, total_frames: int, fps: float,
                       detections: dict, elapsed: float) -> np.ndarray:
    """Crea un pannello informativo laterale."""
    panel_w = 320
    panel_h = 640
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Sfondo scuro

    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Titolo
    cv2.putText(panel, "MARITIME SURVEILLANCE", (10, y), font, 0.6, (0, 200, 255), 2)
    y += 35

    # Separatore
    cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 25

    # Frame info
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 25
    cv2.putText(panel, f"FPS: {fps:.1f}", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 25
    cv2.putText(panel, f"Elapsed: {elapsed:.1f}s", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 35

    # Separatore
    cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 25

    # Detection counts
    cv2.putText(panel, "DETECTIONS", (10, y), font, 0.6, (0, 255, 200), 2)
    y += 30

    for cls_id, cls_name in CLASS_NAMES.items():
        count = detections.get(cls_id, 0)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        # Icona colorata
        cv2.rectangle(panel, (10, y - 12), (25, y + 2), color, -1)
        cv2.putText(panel, f"{cls_name}: {count}", (35, y), font, 0.55, (255, 255, 255), 1)
        y += 30

    total_det = sum(detections.values())
    y += 10
    cv2.putText(panel, f"Total: {total_det}", (10, y), font, 0.55, (0, 200, 255), 1)
    y += 40

    # Separatore
    cv2.line(panel, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 25

    # Status
    cv2.putText(panel, "STATUS: SCANNING", (10, y), font, 0.5, (0, 255, 0), 1)
    y += 25

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(panel, f"Time: {ts}", (10, y), font, 0.4, (150, 150, 150), 1)
    y += 35

    # Barra di progresso
    progress = frame_idx / max(total_frames, 1)
    bar_x = 10
    bar_y = y
    bar_w = panel_w - 20
    bar_h = 15
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 255), -1)
    y += 30
    cv2.putText(panel, f"Progress: {progress * 100:.0f}%", (10, y), font, 0.4, (150, 150, 150), 1)

    return panel


def draw_detections(img: np.ndarray, results) -> tuple[np.ndarray, dict]:
    """Disegna bounding box e label sull'immagine. Ritorna immagine e conteggi."""
    annotated = img.copy()
    counts = defaultdict(int)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cls_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4), font, 0.5, (0, 0, 0), 1)

            counts[cls_id] += 1

    return annotated, dict(counts)


def run_demo(args):
    from ultralytics import YOLO

    model_path = Path(args.model)
    source_dir = Path(args.source)

    if not model_path.exists():
        print(f"❌  Modello non trovato: {model_path}")
        print("   Esegui prima: python scripts/train.py")
        print("   Oppure specifica un modello: --model path/to/best.pt")

        # Fallback: usa un modello pre-trainato per demo
        print("\n🔄  Uso YOLOv8n pre-trainato (COCO) per demo...")
        model = YOLO("yolov8n.pt")
    else:
        print(f"✅  Loading model: {model_path}")
        model = YOLO(str(model_path))

    # Collect images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp")
    images = []
    for ext in extensions:
        images.extend(sorted(glob.glob(str(source_dir / ext))))

    if not images:
        print(f"❌  Nessuna immagine trovata in: {source_dir}")
        return

    print(f"🖼️   Found {len(images)} images in {source_dir}")
    print(f"⏱️   Delay between frames: {args.delay}s")
    print(f"🎯  Confidence threshold: {args.conf}")
    print()

    # Video writer (opzionale)
    video_writer = None
    repeat = 1
    if args.save_video:
        output_path = Path(args.save_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Frame size = image (640) + panel (320)
        # Write each frame multiple times so playback matches the display delay
        fps_out = 1.0  # 1 FPS → each frame lasts 1s in the video
        repeat = max(int(args.delay * fps_out), 1)  # repeat frames to fill delay
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (960, 640))
        print(f"📹  Saving video to: {output_path}")

    # Cumulative stats
    cumulative_detections = defaultdict(int)
    start_time = time.time()

    print("=" * 50)
    print("🚢  YOLO at Sea — Streaming Demo Started")
    print("   Press 'q' to quit | 'n' for next | 'p' to pause")
    print("=" * 50)

    paused = False

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize to 640x640 for display
        img_display = cv2.resize(img, (640, 640))

        # Inference
        t0 = time.time()
        results = model.predict(
            img_path,
            conf=args.conf,
            imgsz=640,
            verbose=False,
        )
        inference_time = time.time() - t0
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Draw detections
        annotated, frame_counts = draw_detections(img_display, results)

        # Update cumulative counts
        for cls_id, count in frame_counts.items():
            cumulative_detections[cls_id] += count

        # Add image name overlay
        img_name = Path(img_path).stem
        cv2.putText(annotated, f"[{img_name}]", (10, 630), cv2.FONT_HERSHEY_SIMPLEX,
                     0.4, (200, 200, 200), 1)

        # Create info panel
        elapsed = time.time() - start_time
        panel = create_info_panel(idx, len(images), fps, frame_counts, elapsed)

        # Combine image + panel
        combined = np.hstack([annotated, panel])

        # Display (only if GUI is available)
        if not HEADLESS:
            cv2.imshow(WINDOW_NAME, combined)

        # Save to video (repeat frame to match display delay)
        if video_writer is not None:
            for _ in range(repeat):
                video_writer.write(combined)

        # Console log
        det_str = ", ".join(f"{CLASS_NAMES.get(k, k)}:{v}" for k, v in frame_counts.items()) or "none"
        print(f"  [{idx:3d}/{len(images)}] {img_name:40s} | FPS: {fps:5.1f} | Det: {det_str}")

        # Wait / key handling
        if HEADLESS:
            time.sleep(args.delay)
        else:
            wait_ms = int(args.delay * 1000)
            while True:
                key = cv2.waitKey(wait_ms if not paused else 100) & 0xFF
                if key == ord("q"):
                    print("\n⏹️  Demo stopped by user")
                    if video_writer:
                        video_writer.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord("n"):
                    break
                elif key == ord("p"):
                    paused = not paused
                    status = "PAUSED" if paused else "RUNNING"
                    print(f"   ⏸️  {status}")
                elif not paused:
                    break

    # Cleanup
    if video_writer:
        video_writer.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊  SESSION SUMMARY")
    print("=" * 50)
    print(f"   Images processed : {len(images)}")
    print(f"   Total time       : {total_time:.1f}s")
    print(f"   Avg time/image   : {total_time / len(images):.2f}s")
    for cls_id, count in sorted(cumulative_detections.items()):
        print(f"   {CLASS_NAMES.get(cls_id, cls_id):15s}: {count}")
    print(f"   Total detections : {sum(cumulative_detections.values())}")
    if args.save_video:
        print(f"   Video saved to   : {args.save_video}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="YOLO at Sea — Real-Time Streaming Demo")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to trained model weights")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE),
                        help="Directory with images to process")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold")
    parser.add_argument("--delay", type=float, default=DISPLAY_DELAY,
                        help="Delay between frames (seconds)")
    parser.add_argument("--save-video", type=str, default=str(DEFAULT_OUTPUT_VIDEO),
                        help="Output video path (empty to disable)")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video saving")
    args = parser.parse_args()

    if args.no_video:
        args.save_video = None

    if HEADLESS and not args.save_video:
        args.save_video = str(DEFAULT_OUTPUT_VIDEO)
        print("ℹ️  Headless mode: enabling video output automatically")

    run_demo(args)


if __name__ == "__main__":
    main()

