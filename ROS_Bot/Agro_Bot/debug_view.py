#!/usr/bin/env python3
"""
debug_view.py — Raw stream viewer using ffmpeg subprocess pipe.
Works on Windows without GStreamer. Requires ffmpeg in PATH.

Install ffmpeg: https://www.gyan.dev/ffmpeg/builds/ (add to PATH)

Usage:
    python debug_view.py --port 5000
"""
import cv2
import numpy as np
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

W, H = args.width, args.height

# ── Try to open with OpenCV FFmpeg first ──────────────────────────────────────
print(f"[INFO] Trying OpenCV FFmpeg backend on UDP port {args.port}...")
urls = [
    f"udp://0.0.0.0:{args.port}",
    f"udp://@:{args.port}",
    f"udp://localhost:{args.port}",
]

cap = None
for url in urls:
    c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if c.isOpened():
        cap = c
        print(f"[INFO] Opened with: {url}")
        break
    c.release()

# ── Fallback: ffmpeg subprocess pipe ─────────────────────────────────────────
if cap is None:
    print("[INFO] OpenCV URL failed. Trying ffmpeg subprocess pipe...")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "quiet",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-rtbufsize", "100M",
        "-i", f"udp://0.0.0.0:{args.port}?overrun_nonfatal=1&fifo_size=5000000",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={W}:{H}",
        "pipe:1"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[INFO] ffmpeg subprocess started. Waiting for frames...")
        frame_size = W * H * 3

        cv2.namedWindow("Debug View (ffmpeg)", cv2.WINDOW_NORMAL)
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                print("[WARN] Stream ended or insufficient data.")
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
            cv2.imshow("Debug View (ffmpeg)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        proc.terminate()
        cv2.destroyAllWindows()
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found in PATH!")
        print("Download from: https://www.gyan.dev/ffmpeg/builds/")
        print("Add the 'bin' folder to your system PATH, then restart terminal.")
    sys.exit(0)

# ── OpenCV loop ───────────────────────────────────────────────────────────────
print("[INFO] Stream open! Press Q to quit.")
cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Waiting for frames...")
        continue
    cv2.imshow("Debug View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
