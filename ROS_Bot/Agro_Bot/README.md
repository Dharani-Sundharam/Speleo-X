# Agro-Bot Vision Pipeline

Distributed low-latency AI vision: Pi 4 (CSI camera) → Laptop RTX 3050.

## Architecture
```
Pi 4 (CSI cam)  ─── H.264/RTP/UDP (5Mbps) ───▶  Laptop RTX 3050
pi_stream.sh                                      laptop_receiver.py
libcamerasrc + x264enc                            OpenCV + YOLOv8 CUDA
640×480 @ 30fps                                   Greenness Index overlay
```

---

## Step 1 — Pi Setup (Ubuntu Server)

SSH into the Pi and run:
```bash
# Enable camera (Ubuntu Server — edit boot config)
sudo bash -c 'echo "camera_auto_detect=1" >> /boot/firmware/config.txt'
sudo reboot  # reboot Pi, then SSH back in

# Install GStreamer + V4L2 tools
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  v4l-utils

# Verify camera shows up
v4l2-ctl --list-devices   # should list /dev/video0
```

### Start streaming (run on Pi):
```bash
./pi_stream.sh <LAPTOP_IP>
# Example: ./pi_stream.sh 192.168.0.105
```

---

## Step 2 — Laptop Setup (Windows)

```bash
# Install Python deps (CUDA 12.1 for RTX 3050)
pip install -r Agro_Bot/requirements.txt

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** OpenCV from pip does NOT include GStreamer by default on Windows.
> You need a GStreamer-enabled build. Options:
> 1. **Recommended:** Install [GStreamer](https://gstreamer.freedesktop.org/download/) (MSVC 64-bit), then install `opencv-python` via pip — it picks up the system GStreamer.
> 2. Or use `pip install opencv-contrib-python` from a wheel with GStreamer support.

### Start receiver (run on Laptop):
```bash
python Agro_Bot/laptop_receiver.py --port 5000 --greenness
```

---

## Controls (while running)
| Key | Action |
|---|---|
| `G` | Toggle Greenness Index overlay |
| `Q` | Quit |

---

## AI Model

YOLOv8n downloads automatically (~6MB) on first run.
To use a custom fine-tuned model, change `YOLO_MODEL` in `laptop_receiver.py`:
```python
YOLO_MODEL = "path/to/your/crop_pest_model.pt"
```

### Fine-tuning on PlantVillage (future step):
```bash
yolo train data=plantvillage.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

## Greenness Index (VDVI)
```
VDVI = (2×G - R - B) / (2×G + R + B)
```
- `> 0.2` → Healthy (shown green)
- `0.0 – 0.2` → Moderate stress (shown yellow)
- `< 0.0` → Stressed / diseased (shown red)
