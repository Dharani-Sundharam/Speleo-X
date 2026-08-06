#!/usr/bin/env python3
"""
speleo_dashboard.py  —  Speleo-X Web Control Dashboard
=======================================================
Serves a browser-based robot control dashboard on port 8080.
No ROS. No Foxglove. Just open a browser on any device on the same WiFi.

Install:
    pip3 install aiohttp pyserial

Run:
    python3 speleo_dashboard.py

Then open:
    http://<PI_IP>:8080

Controls:
    Browser W/A/S/D or on-screen buttons → motor commands over serial
    Live telemetry: encoder ticks, IMU (accel + gyro), odometry

WebSocket protocol  (JSON, text frames)
    Server → Browser:
        {"type":"telemetry","t":1234,"enc_l":22,"enc_r":2,
         "ax":-0.06,"ay":0.41,"az":9.98,"gx":0.0,"gy":0.0,"gz":0.0,
         "x":0.0,"y":0.0,"th":0.0,"vx":0.0}
    Browser → Server:
        {"type":"cmd","left":200,"right":200}
        {"type":"stop"}
"""

import asyncio
import json
import math
import queue
import serial
import threading
import time
import os
from aiohttp import web

# ── Configuration ─────────────────────────────────────────────────────────────
HOST            = "0.0.0.0"
PORT            = 5000
SERIAL_PORTS    = ["/dev/bluepill", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
BAUD_RATE       = 115200

WHEEL_RADIUS    = 0.035
WHEEL_BASE      = 0.20
TICKS_PER_REV_L = 725
TICKS_PER_REV_R = 711
ACCEL_SCALE     = 16384.0
GYRO_SCALE      = 131.0
GRAVITY         = 9.81

WATCHDOG_SEC    = 1.0   # stop motors if browser goes silent
# ─────────────────────────────────────────────────────────────────────────────

# Shared state
_clients:   set   = set()
_clients_lock     = None   # created inside on_startup (needs running event loop)
_serial_q         = queue.SimpleQueue()
_telemetry: dict  = {}
_ser              = None
_last_cmd_t       = time.time()

# Odometry state
_x = _y = _th = _vx = _wz = 0.0
_prev_enc_l = _prev_enc_r = _prev_t = None


# =============================================================================
# Serial helpers
# =============================================================================

def open_serial():
    for p in SERIAL_PORTS:
        if os.path.exists(p):
            try:
                s = serial.Serial(p, BAUD_RATE, timeout=0.1)
                print(f"[serial] Opened {p}")
                return s
            except serial.SerialException as e:
                print(f"[serial] {p}: {e}")
    print("[serial] WARNING — no STM32 found, running in demo mode")
    return None


def send_motor(left: int, right: int):
    global _last_cmd_t
    _last_cmd_t = time.time()
    if _ser:
        try:
            _ser.write(f"m,{left},{right}\n".encode())
        except Exception as e:
            print(f"[serial] write error: {e}")


# =============================================================================
# Serial reader thread  — runs forever, pushes parsed telemetry
# =============================================================================

def serial_reader():
    global _x, _y, _th, _vx, _wz, _prev_enc_l, _prev_enc_r, _prev_t, _telemetry

    buf = b""
    while True:
        if _ser is None:
            # Demo mode: generate fake telemetry
            _telemetry = {
                "t": int(time.time() * 1000),
                "enc_l": 0, "enc_r": 0,
                "ax": 0.0, "ay": 0.0, "az": GRAVITY,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "x": 0.0, "y": 0.0, "th": 0.0, "vx": 0.0,
                "demo": True,
            }
            time.sleep(0.05)
            continue

        try:
            chunk = _ser.read(max(1, _ser.in_waiting))
            if chunk:
                buf += chunk
                while b"\n" in buf:
                    line_b, buf = buf.split(b"\n", 1)
                    line = line_b.decode("utf-8", errors="ignore").strip()
                    if not line or line.startswith("#"):
                        continue
                    _parse_line(line)
        except Exception as e:
            print(f"[serial] read error: {e}")
            time.sleep(0.1)


def _parse_line(line: str):
    global _x, _y, _th, _vx, _wz, _prev_enc_l, _prev_enc_r, _prev_t, _telemetry
    try:
        fields = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = int(v)

        if not {"ENC_L","ENC_R","AX","AY","AZ","GX","GY","GZ"}.issubset(fields):
            return

        enc_l = fields["ENC_L"]
        enc_r = fields["ENC_R"]
        ax = fields["AX"] / ACCEL_SCALE * GRAVITY
        ay = fields["AY"] / ACCEL_SCALE * GRAVITY
        az = fields["AZ"] / ACCEL_SCALE * GRAVITY
        gx = math.radians(fields["GX"] / GYRO_SCALE)
        gy = math.radians(fields["GY"] / GYRO_SCALE)
        gz = math.radians(fields["GZ"] / GYRO_SCALE)

        now = time.time()
        if _prev_enc_l is not None and _prev_t is not None:
            dl = (enc_l - _prev_enc_l) / TICKS_PER_REV_L * 2 * math.pi * WHEEL_RADIUS
            dr = (enc_r - _prev_enc_r) / TICKS_PER_REV_R * 2 * math.pi * WHEEL_RADIUS
            dc = (dl + dr) / 2.0
            dt_val = (dr - dl) / WHEEL_BASE
            _th  = math.atan2(math.sin(_th + dt_val), math.cos(_th + dt_val))
            _x  += dc * math.cos(_th)
            _y  += dc * math.sin(_th)
            dt   = now - _prev_t
            _vx  = dc / dt if dt > 0 else 0.0
            _wz  = gz

        _prev_enc_l, _prev_enc_r, _prev_t = enc_l, enc_r, now

        _telemetry = {
            "t": fields.get("T", 0),
            "enc_l": enc_l, "enc_r": enc_r,
            "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3),
            "gx": round(gx, 4), "gy": round(gy, 4), "gz": round(gz, 4),
            "x": round(_x, 4), "y": round(_y, 4),
            "th": round(math.degrees(_th), 1),
            "vx": round(_vx, 4),
        }
    except Exception as e:
        pass


# =============================================================================
# Watchdog thread
# =============================================================================

def watchdog():
    prev_pwm = (0, 0)
    while True:
        time.sleep(0.2)
        if time.time() - _last_cmd_t > WATCHDOG_SEC and prev_pwm != (0, 0):
            send_motor(0, 0)
            prev_pwm = (0, 0)
            print("[watchdog] Motors stopped")


# =============================================================================
# Broadcast telemetry to all WebSocket clients  (~20 Hz)
# =============================================================================

async def broadcaster(app):
    global _clients
    while True:
        await asyncio.sleep(0.05)
        if not _telemetry:
            continue
        msg = json.dumps({"type": "telemetry", **_telemetry})
        async with _clients_lock:
            dead = set()
            for ws in _clients:
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.add(ws)
            _clients -= dead


# =============================================================================
# WebSocket handler
# =============================================================================

async def ws_handler(request):
    global _clients
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async with _clients_lock:
        _clients.add(ws)
    print(f"[ws] Client connected: {request.remote}")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("type") == "cmd":
                        left  = int(data.get("left",  0))
                        right = int(data.get("right", 0))
                        left  = max(-255, min(255, left))
                        right = max(-255, min(255, right))
                        send_motor(left, right)
                    elif data.get("type") == "stop":
                        send_motor(0, 0)
                except Exception as e:
                    print(f"[ws] msg error: {e}")
    finally:
        async with _clients_lock:
            _clients.discard(ws)
        print(f"[ws] Client disconnected: {request.remote}")
    return ws


# =============================================================================
# HTML Dashboard
# =============================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Speleo-X Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080c10;
    --bg2:      #0d1117;
    --border:   #1c2333;
    --accent:   #00e5b4;
    --accent2:  #00a886;
    --warn:     #f5a623;
    --danger:   #ff4444;
    --text:     #c9d1d9;
    --dim:      #6e7681;
    --font:     'Space Mono', monospace;
  }

  html, body {
    height: 100%; background: var(--bg); color: var(--text);
    font-family: var(--font); font-size: 13px; overflow: hidden;
  }

  /* ── Top bar ── */
  #topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }
  #topbar .brand { color: var(--accent); font-weight: 700; font-size: 15px; letter-spacing: 3px; }
  #topbar .brand span { color: var(--dim); font-weight: 400; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--danger);
    display: inline-block; margin-right: 8px; transition: background .3s; }
  #status-dot.ok { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  #status-text { font-size: 11px; color: var(--dim); }

  /* ── Main grid ── */
  #main {
    display: grid;
    grid-template-columns: 220px 1fr;
    grid-template-rows: 1fr 180px;
    height: calc(100vh - 45px);
    gap: 1px; background: var(--border);
  }

  .panel {
    background: var(--bg2); padding: 16px; overflow: hidden;
  }
  .panel-title {
    font-size: 10px; letter-spacing: 2px; color: var(--accent2);
    text-transform: uppercase; margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  /* ── Telemetry ── */
  #telemetry { grid-row: 1 / 2; grid-column: 1 / 2; }

  .telem-group { margin-bottom: 14px; }
  .telem-label { font-size: 9px; color: var(--dim); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
  .telem-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
  .telem-key { color: var(--dim); font-size: 11px; }
  .telem-val { color: var(--accent); font-size: 12px; font-weight: 700; }
  .telem-val.warn { color: var(--warn); }

  .bar-wrap { height: 3px; background: var(--border); border-radius: 2px; margin-top: 4px; }
  .bar-fill { height: 3px; background: var(--accent); border-radius: 2px; transition: width .1s; }

  /* ── Radar / LiDAR ── */
  #radar-panel { grid-row: 1 / 2; grid-column: 2 / 3; position: relative; }
  #radar-canvas { display: block; margin: 0 auto; }
  #no-lidar {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    text-align: center; color: var(--dim);
  }
  #no-lidar .big { font-size: 32px; margin-bottom: 8px; opacity: .3; }
  #no-lidar .msg { font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }

  /* ── Controls ── */
  #controls { grid-column: 1 / 3; display: flex; align-items: center;
    justify-content: center; gap: 40px; }

  .dpad { display: grid; grid-template-columns: repeat(3, 56px); grid-template-rows: repeat(3, 56px); gap: 4px; }
  .btn {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; font-family: var(--font); font-size: 18px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background .1s, border-color .1s, color .1s, box-shadow .1s;
    user-select: none; -webkit-user-select: none;
  }
  .btn:active, .btn.pressed {
    background: var(--accent2); border-color: var(--accent);
    color: var(--bg); box-shadow: 0 0 12px var(--accent2);
  }
  .btn.stop-btn {
    grid-column: 2; grid-row: 2;
    border-color: var(--danger); color: var(--danger); font-size: 12px;
  }
  .btn.stop-btn:active, .btn.stop-btn.pressed {
    background: var(--danger); color: #fff; box-shadow: 0 0 12px var(--danger);
  }

  /* speed slider */
  .speed-wrap { text-align: center; }
  .speed-wrap label { display: block; font-size: 10px; letter-spacing: 1px;
    color: var(--dim); text-transform: uppercase; margin-bottom: 8px; }
  #speed-val { color: var(--accent); font-weight: 700; }
  input[type=range] {
    -webkit-appearance: none; appearance: none;
    width: 120px; height: 4px; background: var(--border); border-radius: 2px; outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: var(--accent); cursor: pointer; border: 2px solid var(--bg);
    box-shadow: 0 0 6px var(--accent);
  }

  /* heading compass */
  #heading-wrap { text-align: center; }
  #compass {
    width: 80px; height: 80px; margin: 0 auto 6px;
    border: 1px solid var(--border); border-radius: 50%; position: relative;
    background: var(--bg);
  }
  #compass-needle {
    position: absolute; top: 50%; left: 50%;
    width: 2px; height: 32px; background: var(--accent);
    transform-origin: bottom center;
    transform: translateX(-50%) translateY(-100%) rotate(0deg);
    transition: transform .1s;
    box-shadow: 0 0 6px var(--accent);
  }
  #heading-val { font-size: 11px; color: var(--dim); }
  #heading-val span { color: var(--accent); font-weight: 700; }

  @media (max-width: 600px) {
    #main { grid-template-columns: 1fr; grid-template-rows: auto 280px 180px; }
    #telemetry { grid-row: 1; grid-column: 1; }
    #radar-panel { grid-row: 2; grid-column: 1; }
    #controls { grid-row: 3; grid-column: 1; }
  }
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <div class="brand">SPELEO<span>-X</span> &nbsp;ROVER CONTROL</div>
  <div>
    <span id="status-dot"></span>
    <span id="status-text">DISCONNECTED</span>
  </div>
</div>

<!-- Main grid -->
<div id="main">

  <!-- Telemetry panel -->
  <div class="panel" id="telemetry">
    <div class="panel-title">Telemetry</div>

    <div class="telem-group">
      <div class="telem-label">Odometry</div>
      <div class="telem-row"><span class="telem-key">X</span><span class="telem-val" id="t-x">0.000 m</span></div>
      <div class="telem-row"><span class="telem-key">Y</span><span class="telem-val" id="t-y">0.000 m</span></div>
      <div class="telem-row"><span class="telem-key">Heading</span><span class="telem-val" id="t-th">0.0°</span></div>
      <div class="telem-row"><span class="telem-key">Speed</span><span class="telem-val" id="t-vx">0.000 m/s</span></div>
    </div>

    <div class="telem-group">
      <div class="telem-label">Encoders</div>
      <div class="telem-row"><span class="telem-key">Left</span><span class="telem-val" id="t-encl">0</span></div>
      <div class="telem-row"><span class="telem-key">Right</span><span class="telem-val" id="t-encr">0</span></div>
    </div>

    <div class="telem-group">
      <div class="telem-label">Accelerometer (m/s²)</div>
      <div class="telem-row"><span class="telem-key">X</span><span class="telem-val" id="t-ax">0.00</span></div>
      <div class="telem-row"><span class="telem-key">Y</span><span class="telem-val" id="t-ay">0.00</span></div>
      <div class="telem-row"><span class="telem-key">Z</span><span class="telem-val" id="t-az">0.00</span></div>
      <div class="bar-wrap"><div class="bar-fill" id="az-bar" style="width:50%"></div></div>
    </div>

    <div class="telem-group">
      <div class="telem-label">Gyroscope (rad/s)</div>
      <div class="telem-row"><span class="telem-key">Z (yaw)</span><span class="telem-val" id="t-gz">0.0000</span></div>
    </div>
  </div>

  <!-- Radar panel -->
  <div class="panel" id="radar-panel">
    <div class="panel-title">LiDAR Scan</div>
    <canvas id="radar-canvas"></canvas>
    <div id="no-lidar">
      <div class="big">⬡</div>
      <div class="msg">LiDAR Not Connected</div>
      <div style="font-size:10px;color:var(--dim);margin-top:6px">Add YDLiDAR to enable scan</div>
    </div>
  </div>

  <!-- Controls panel -->
  <div class="panel" id="controls">

    <!-- D-Pad -->
    <div class="dpad">
      <div></div>
      <button class="btn" id="btn-fwd"  data-key="w">▲</button>
      <div></div>
      <button class="btn" id="btn-left" data-key="a">◀</button>
      <button class="btn stop-btn" id="btn-stop" data-key=" ">■</button>
      <button class="btn" id="btn-right" data-key="d">▶</button>
      <div></div>
      <button class="btn" id="btn-rev"  data-key="s">▼</button>
      <div></div>
    </div>

    <!-- Speed -->
    <div class="speed-wrap">
      <label>Speed  <span id="speed-val">200</span></label>
      <input type="range" id="speed" min="50" max="255" value="200" step="5">
    </div>

    <!-- Compass -->
    <div id="heading-wrap">
      <div id="compass"><div id="compass-needle"></div></div>
      <div id="heading-val">Heading: <span id="h-deg">0.0</span>°</div>
    </div>

  </div>
</div>

<script>
// ── WebSocket ──────────────────────────────────────────────────────────────
const statusDot  = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
let ws, reconnectTimer;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    statusDot.classList.add('ok');
    statusText.textContent = 'CONNECTED  ●  ' + location.host;
    clearTimeout(reconnectTimer);
  };

  ws.onclose = ws.onerror = () => {
    statusDot.classList.remove('ok');
    statusText.textContent = 'RECONNECTING...';
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.type === 'telemetry') updateTelemetry(d);
    if (d.type === 'lidar')     drawLidar(d.points);
  };
}

function sendCmd(left, right) {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({type:'cmd', left, right}));
}
function sendStop() {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({type:'stop'}));
}

connect();

// ── Telemetry ──────────────────────────────────────────────────────────────
function updateTelemetry(d) {
  setText('t-x',    d.x.toFixed(3) + ' m');
  setText('t-y',    d.y.toFixed(3) + ' m');
  setText('t-th',   d.th.toFixed(1) + '°');
  setText('t-vx',   d.vx.toFixed(3) + ' m/s');
  setText('t-encl', d.enc_l);
  setText('t-encr', d.enc_r);
  setText('t-ax',   d.ax.toFixed(2));
  setText('t-ay',   d.ay.toFixed(2));
  setText('t-az',   d.az.toFixed(2));
  setText('t-gz',   d.gz.toFixed(4));

  // AZ bar (gravity indicator)
  const azPct = Math.min(100, Math.max(0, (d.az / 15) * 100));
  document.getElementById('az-bar').style.width = azPct + '%';

  // Compass needle
  document.getElementById('compass-needle').style.transform =
    `translateX(-50%) translateY(-100%) rotate(${d.th}deg)`;
  document.getElementById('h-deg').textContent = d.th.toFixed(1);
}
function setText(id, val) { document.getElementById(id).textContent = val; }

// ── Radar canvas ───────────────────────────────────────────────────────────
const canvas = document.getElementById('radar-canvas');
const ctx    = canvas.getContext('2d');
let lidarPoints = [];
let sweepAngle  = 0;

function resizeRadar() {
  const panel  = document.getElementById('radar-panel');
  const size   = Math.min(panel.clientWidth - 32, panel.clientHeight - 50);
  canvas.width  = size;
  canvas.height = size;
}
resizeRadar();
window.addEventListener('resize', resizeRadar);

function drawRadar() {
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2, R = W / 2 - 8;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#080c10';
  ctx.fillRect(0, 0, W, H);

  // Grid rings
  const rings = 4;
  for (let i = 1; i <= rings; i++) {
    const r = (R / rings) * i;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = '#1c2333';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#6e7681';
    ctx.font = '9px Space Mono, monospace';
    ctx.fillText(`${(i * 1).toFixed(0)}m`, cx + 3, cy - r + 12);
  }

  // Cross-hairs
  ctx.strokeStyle = '#1c2333';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.stroke();

  // Sweep gradient
  const sweepGrad = ctx.createConicalGradient
    ? ctx.createConicalGradient(cx, cy, sweepAngle)
    : null;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(sweepAngle);
  const sweepG = ctx.createLinearGradient(0, 0, R, 0);
  sweepG.addColorStop(0,   'rgba(0,229,180,0.25)');
  sweepG.addColorStop(1,   'rgba(0,229,180,0)');
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.arc(0, 0, R, -Math.PI / 6, 0);
  ctx.fillStyle = sweepG;
  ctx.fill();
  ctx.restore();

  // Sweep line
  ctx.save();
  ctx.translate(cx, cy);
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(R * Math.cos(sweepAngle), R * Math.sin(sweepAngle));
  ctx.strokeStyle = 'rgba(0,229,180,0.8)';
  ctx.lineWidth   = 1.5;
  ctx.stroke();
  ctx.restore();

  sweepAngle = (sweepAngle + 0.04) % (Math.PI * 2);

  // LiDAR points
  const maxDist = 4.0; // metres
  for (const [angleDeg, distMm] of lidarPoints) {
    const distM = distMm / 1000;
    if (distM <= 0 || distM > maxDist) continue;
    const a   = (angleDeg - 90) * Math.PI / 180;
    const pct = distM / maxDist;
    const px  = cx + Math.cos(a) * pct * R;
    const py  = cy + Math.sin(a) * pct * R;
    const alpha = Math.max(0.2, 1 - pct);
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0,229,180,${alpha})`;
    ctx.fill();
  }

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#00e5b4';
  ctx.fill();

  requestAnimationFrame(drawRadar);
}
drawRadar();

function drawLidar(points) {
  lidarPoints = points;
  document.getElementById('no-lidar').style.display = 'none';
}

// ── Controls ───────────────────────────────────────────────────────────────
const speed = () => parseInt(document.getElementById('speed').value);
document.getElementById('speed').addEventListener('input', e => {
  document.getElementById('speed-val').textContent = e.target.value;
});

const KEY_MAP = {
  'w': () => sendCmd( speed(),  speed()),
  's': () => sendCmd(-speed(), -speed()),
  'a': () => sendCmd(-speed(),  speed()),
  'd': () => sendCmd( speed(), -speed()),
  ' ': () => sendStop(),
};
const STOP_ON_RELEASE = new Set(['w','s','a','d']);

// ── Continuous hold: re-send command every 250ms while key held ────────────
let _cmdInterval = null;
let _activeCmd   = null;

function startHold(key) {
  if (_activeCmd === key) return;          // already holding this key
  stopHold();                              // cancel any previous hold
  _activeCmd = key;
  if (KEY_MAP[key]) {
    KEY_MAP[key]();                        // send immediately
    if (key !== ' ') {
      _cmdInterval = setInterval(() => {
        if (KEY_MAP[_activeCmd]) KEY_MAP[_activeCmd]();
      }, 250);                             // re-send every 250ms (watchdog=1s)
    }
  }
  highlightBtn(key, true);
}

function stopHold(key) {
  if (key && key !== _activeCmd) return;  // ignore if different key
  clearInterval(_cmdInterval);
  _cmdInterval = null;
  if (_activeCmd && STOP_ON_RELEASE.has(_activeCmd)) sendStop();
  highlightBtn(_activeCmd, false);
  _activeCmd = null;
}

const held = new Set();
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (KEY_MAP[k] && !held.has(k)) { held.add(k); startHold(k); }
  if (k === ' ') { e.preventDefault(); sendStop(); }
});
document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  held.delete(k);
  stopHold(k);
});

function highlightBtn(key, on) {
  document.querySelectorAll('.btn[data-key]').forEach(b => {
    if (b.dataset.key === key) b.classList.toggle('pressed', on);
  });
}

// On-screen buttons — same interval pattern
document.querySelectorAll('.btn').forEach(btn => {
  const key = btn.dataset.key;
  btn.addEventListener('mousedown',  ()  => startHold(key));
  btn.addEventListener('touchstart', e  => { e.preventDefault(); startHold(key); }, {passive:false});
  btn.addEventListener('mouseup',    ()  => stopHold(key));
  btn.addEventListener('touchend',   ()  => stopHold(key));
  btn.addEventListener('mouseleave', ()  => stopHold(key));
});

</script>
</body>
</html>"""


# =============================================================================
# aiohttp routes
# =============================================================================

async def index(request):
    return web.Response(text=HTML, content_type="text/html")


# =============================================================================
# App startup / shutdown
# =============================================================================

async def on_startup(app):
    global _ser, _clients_lock
    _clients_lock = asyncio.Lock()   # create inside running event loop
    _ser = open_serial()
    threading.Thread(target=serial_reader, daemon=True).start()
    threading.Thread(target=watchdog,      daemon=True).start()
    asyncio.create_task(broadcaster(app))
    print(f"\n{'='*50}")
    print(f"  Speleo-X Dashboard  →  http://<PI_IP>:{PORT}")
    print(f"{'='*50}\n")


app = web.Application()
app.router.add_get("/",   index)
app.router.add_get("/ws", ws_handler)
app.on_startup.append(on_startup)


if __name__ == "__main__":
    web.run_app(app, host=HOST, port=PORT)
