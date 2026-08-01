#!/usr/bin/env python3
"""
laptop_receiver.py — Agro-Bot AI Vision Processor
Receives MPEG-TS/H.264 stream from Pi via ffmpeg pipe.
Runs YOLOv8 (CUDA) + Greenness Index overlay.

Usage:
    python laptop_receiver.py [--port 5000] [--greenness]

Requirements:
    pip install -r requirements.txt
    ffmpeg in PATH (https://www.gyan.dev/ffmpeg/builds/)
"""

import cv2
import numpy as np
import subprocess
import argparse
import time
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[WARN] ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
W, H        = 640, 480
YOLO_MODEL  = r"runs\detect\Agro_Bot\runs\plant_detection4\weights\best.pt"  # trained plant model
CONF        = 0.4
DEVICE      = "cuda"
GI_ALPHA    = 0.4


# ── Greenness Index (VDVI) ────────────────────────────────────────────────────
def greenness_overlay(frame):
    f = frame.astype(np.float32) + 1e-6
    B, G, R = f[:,:,0], f[:,:,1], f[:,:,2]
    vdvi = (2*G - R - B) / (2*G + R + B)
    vdvi_norm = np.clip((vdvi + 1) / 2, 0, 1)   # normalize -1..+1 to 0..1
    # Manual Red→Yellow→Green gradient (cv2.COLORMAP_RdYlGn doesn't exist in OpenCV)
    r = np.clip(2.0 * (1.0 - vdvi_norm), 0, 1)
    g = np.clip(2.0 * vdvi_norm,          0, 1)
    b = np.zeros_like(r)
    coloured = (np.stack([b, g, r], axis=-1) * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame, 1 - GI_ALPHA, coloured, GI_ALPHA, 0)
    return overlay, float(vdvi_norm.mean())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      type=int,            default=5000)
    parser.add_argument("--greenness", action="store_true", help="Start with greenness overlay on")
    args = parser.parse_args()

    # Load YOLOv8
    model = None
    if YOLO_AVAILABLE:
        print(f"[INFO] Loading {YOLO_MODEL} on {DEVICE}...")
        model = YOLO(YOLO_MODEL)
        model.to(DEVICE)

    # Start ffmpeg pipe
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "quiet",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-rtbufsize", "100M",
        "-i", f"udp://0.0.0.0:{args.port}?overrun_nonfatal=1&fifo_size=5000000",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={W}:{H}",
        "pipe:1"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found! Download: https://www.gyan.dev/ffmpeg/builds/")
        sys.exit(1)

    print(f"[INFO] Listening on UDP port {args.port}... Press Q to quit, G to toggle greenness.")
    frame_size = W * H * 3
    show_gi = args.greenness
    fps_t = time.time()
    frames = 0
    fps = 0.0

    cv2.namedWindow("Agro-Bot Vision", cv2.WINDOW_NORMAL)

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            print("[WARN] Stream ended.")
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))

        # ── YOLOv8 inference ───────────────────────────────────────────────
        if model:
            results = model(frame, conf=CONF, device=DEVICE, verbose=False)
            display = results[0].plot()
        else:
            display = frame.copy()

        # ── Greenness Index ────────────────────────────────────────────────
        health = None
        if show_gi:
            display, health = greenness_overlay(display)

        # ── FPS counter ────────────────────────────────────────────────────
        frames += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps = frames / elapsed
            fps_t, frames = time.time(), 0

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if health is not None:
            pct = health * 100
            col = (0,255,0) if pct>50 else (0,165,255) if pct>30 else (0,0,255)
            cv2.putText(display, f"Crop Health: {pct:.1f}%", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        cv2.imshow("Agro-Bot Vision", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_gi = not show_gi
            print(f"[INFO] Greenness: {'ON' if show_gi else 'OFF'}")

    proc.terminate()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
