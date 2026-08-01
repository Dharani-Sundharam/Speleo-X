#!/bin/bash
# pi_stream.sh — Agro-Bot Pi 4 Camera Sender
# Streams 640x480 H.264 video over UDP to the laptop at ~5-10 Mbps
# Uses libcamerasrc (Ubuntu Server / Pi OS with libcamera) + x264enc
#
# Usage:
#   chmod +x pi_stream.sh
#   ./pi_stream.sh <LAPTOP_IP>
#
# Install requirements on Pi (Ubuntu Server):
#   sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-good \
#     gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
#     gstreamer1.0-libav libgstreamer1.0-dev \
#     libcamera-tools gstreamer1.0-libcamera

LAPTOP_IP="${10.17.144.64}"   # Pass laptop IP as first argument
PORT=5000
WIDTH=640
HEIGHT=480
FPS=30
BITRATE=4000   # kbps — increase for higher quality, decrease to reduce latency

echo "Streaming ${WIDTH}x${HEIGHT} @ ${FPS}fps → ${LAPTOP_IP}:${PORT}"
echo "Press Ctrl+C to stop."

gst-launch-1.0 -e \
  v4l2src device=/dev/video0 ! \
  video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1 ! \
  videoconvert ! \
  video/x-raw,format=I420 ! \
  x264enc \
    bitrate=${BITRATE} \
    tune=zerolatency \
    speed-preset=ultrafast \
    key-int-max=30 ! \
  mpegtsmux ! \
  udpsink host=${LAPTOP_IP} port=${PORT} sync=false async=false
