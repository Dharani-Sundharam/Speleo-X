#!/usr/bin/env python3
"""
speleo_dashboard.py  —  Speleo-X Robot Control Dashboard v2
=============================================================
Self-hosted cave robot control dashboard. Open in any browser on the same network.

Features:
  - BreezySlam SLAM map (built from YDLiDAR + wheel odometry)
  - Live IMU + encoder telemetry
  - Continuous WASD keyboard + on-screen controls
  - Apple/Google-style clean light UI

Install:
    pip3 install aiohttp pyserial breezyslam --break-system-packages

Run:
    python3 speleo_dashboard.py

Open:
    http://<PI_IP>:5000
"""

import asyncio
import base64
import json
import math
import queue
import serial
import threading
import time
import os
from aiohttp import web

# ── Configuration ─────────────────────────────────────────────────────────────
HOST             = "0.0.0.0"
PORT             = 5000
SERIAL_PORTS     = ["/dev/bluepill", "/dev/ttyUSB0", "/dev/ttyACM0"]
BAUD_RATE        = 115200

WHEEL_RADIUS     = 0.035
WHEEL_BASE       = 0.20
TICKS_PER_REV_L  = 725
TICKS_PER_REV_R  = 711
ACCEL_SCALE      = 16384.0
GYRO_SCALE       = 131.0
GRAVITY          = 9.81
WATCHDOG_SEC     = 1.0

LIDAR_PORT       = "/dev/ttyUSB1"
LIDAR_BAUD       = 115200
LIDAR_FREQ       = 10.0
LIDAR_SAMPLE_RATE= 3
LIDAR_SINGLE_CH  = True
LIDAR_REVERSION  = True
LIDAR_INVERTED   = True
LIDAR_MIN_RANGE  = 0.01    # metres
LIDAR_MAX_RANGE  = 12.0    # metres
LIDAR_SCAN_SIZE  = 280     # "Single Fixed Size: 280" from SDK output
LIDAR_ANGLE_OFFSET_DEG = 180.0  # rotate LiDAR frame to match robot forward direction
                                  # (0=no rotation; adjust if obstacles appear backwards)

MAP_SIZE_PIXELS  = 512
MAP_SIZE_METERS  = 25.0
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared state ──────────────────────────────────────────────────────────────
_clients:   set  = set()
_clients_lock    = None
_lidar_queue     = queue.SimpleQueue()
_map_queue       = queue.SimpleQueue()
_telemetry: dict = {}
_ser             = None
_last_cmd_t      = time.time()
_serial_status   = "demo"
_parse_count     = 0
_lidar_status    = "offline"

# Odometry — starts at zero every run; _prev_enc_l=None means first reading
_x = _y = _th = _vx = 0.0
_prev_enc_l = _prev_enc_r = _prev_t = None

# SLAM
_slam_lock       = threading.Lock()
_slam_obj        = None        # RMHC_SLAM instance, False if unavailable
_slam_pose       = [0.0, 0.0, 0.0]   # [x_mm, y_mm, th_deg]
_last_slam_odom  = None        # (x_mm, y_mm, th_deg, time)
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Serial helpers
# =============================================================================

def open_serial():
    for p in SERIAL_PORTS:
        if os.path.exists(p):
            try:
                s = serial.Serial(p, BAUD_RATE, timeout=0.1)
                print(f"[serial] Opened {p} @ {BAUD_RATE}")
                return s
            except serial.SerialException as e:
                print(f"[serial] {p}: {e}")
    print("[serial] No STM32 found — running in demo mode")
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
# Serial reader thread
# =============================================================================

def serial_reader():
    global _x, _y, _th, _vx
    global _prev_enc_l, _prev_enc_r, _prev_t
    global _telemetry, _serial_status, _parse_count

    buf = b""
    while True:
        if _ser is None:
            _serial_status = "demo"
            _telemetry = {
                "t": int(time.time() * 1000),
                "enc_l": 0, "enc_r": 0,
                "ax": 0.0, "ay": 0.0, "az": GRAVITY,
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "x": 0.0, "y": 0.0, "th": 0.0, "vx": 0.0,
                "serial": "demo",
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
                    if line:
                        _parse_line(line)
        except Exception as e:
            print(f"[serial] read error: {e}")
            time.sleep(0.1)


def _parse_line(line: str):
    global _x, _y, _th, _vx
    global _prev_enc_l, _prev_enc_r, _prev_t
    global _telemetry, _serial_status, _parse_count

    try:
        fields = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = int(v)

        if not {"ENC_L", "ENC_R", "AX", "AY", "AZ", "GX", "GY", "GZ"}.issubset(fields):
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
            dth = (dr - dl) / WHEEL_BASE
            _th = math.atan2(math.sin(_th + dth), math.cos(_th + dth))
            _x += dc * math.cos(_th)
            _y += dc * math.sin(_th)
            dt = now - _prev_t
            _vx = dc / dt if dt > 0 else 0.0

        _prev_enc_l, _prev_enc_r, _prev_t = enc_l, enc_r, now

        if _serial_status != "live":
            _serial_status = "live"
            print(f"[serial] ✓ Live telemetry: ENC_L={enc_l} ENC_R={enc_r} AZ={az:.2f}m/s²")

        _parse_count += 1
        if _parse_count % 100 == 0:
            print(f"[telem]  ENC_L={enc_l:7d}  ENC_R={enc_r:7d} | "
                  f"AZ={az:6.2f} GZ={gz:+.4f} | "
                  f"X={_x:.3f}m Y={_y:.3f}m TH={math.degrees(_th):.1f}°")

        _telemetry = {
            "t":      fields.get("T", 0),
            "enc_l":  enc_l,
            "enc_r":  enc_r,
            "ax":     round(ax, 3),
            "ay":     round(ay, 3),
            "az":     round(az, 3),
            "gx":     round(gx, 4),
            "gy":     round(gy, 4),
            "gz":     round(gz, 4),
            "x":      round(_x, 4),
            "y":      round(_y, 4),
            "th":     round(math.degrees(_th), 2),
            "vx":     round(_vx, 4),
            "serial": "live",
        }
    except Exception as e:
        print(f"[parse] error on '{line[:60]}': {e}")


# =============================================================================
# Watchdog thread
# =============================================================================

def watchdog():
    prev = (0, 0)
    while True:
        time.sleep(0.2)
        if time.time() - _last_cmd_t > WATCHDOG_SEC and prev != (0, 0):
            send_motor(0, 0)
            prev = (0, 0)


# =============================================================================
# SLAM helpers
# =============================================================================

def _init_slam():
    global _slam_obj, _slam_update_mode
    try:
        from breezyslam.algorithms import RMHC_SLAM
        from breezyslam.sensors import Laser
        laser = Laser(
            scan_size=LIDAR_SCAN_SIZE,
            scan_rate_hz=LIDAR_FREQ,
            detection_angle_degrees=360,
            distance_no_detection_mm=int(LIDAR_MAX_RANGE * 1000),
            detection_margin=0,
            offset_mm=0,
        )
        _slam_obj = RMHC_SLAM(laser, MAP_SIZE_PIXELS, MAP_SIZE_METERS, random_seed=42)
        print(f"[slam] ✓ BreezySlam initialized  ({MAP_SIZE_PIXELS}x{MAP_SIZE_PIXELS} @ {MAP_SIZE_METERS}m)")

        # ── Probe update() signature with a real dummy velocity tuple ─────────────────
        dummy_scan = [int(LIDAR_MAX_RANGE * 1000)] * LIDAR_SCAN_SIZE
        dummy_vel  = (0.0, 0.0, 0.1)
        for mode in ('kw', 'pos', 'bare'):
            try:
                if   mode == 'kw':  _slam_obj.update(dummy_scan, velocities=dummy_vel)
                elif mode == 'pos': _slam_obj.update(dummy_scan, dummy_vel)
                else:               _slam_obj.update(dummy_scan)
                _slam_update_mode = mode
                print(f"[slam] update() API mode detected: '{mode}'")
                break
            except TypeError:
                continue
        else:
            _slam_update_mode = 'bare'
            print("[slam] update() bare mode (no odometry velocities)")

    except ImportError:
        print("[slam] BreezySlam not installed — using raw LiDAR overlay")
        _slam_obj = False


def _preprocess_scan(points):
    """Convert [(angle_deg, dist_mm)] to uniform LIDAR_SCAN_SIZE array."""
    no_det = int(LIDAR_MAX_RANGE * 1000)
    scan   = [no_det] * LIDAR_SCAN_SIZE
    bw     = 360.0 / LIDAR_SCAN_SIZE
    for angle_deg, dist_mm in points:
        if dist_mm <= 0:
            continue
        idx = int((angle_deg % 360.0) / bw) % LIDAR_SCAN_SIZE
        d   = int(dist_mm)
        if d < scan[idx]:
            scan[idx] = d
    return scan


# ── update() call mode — detected in _init_slam() ────────────────────────────
_slam_update_mode = 'bare'


def _slam_update(points):
    global _slam_obj, _slam_pose, _last_slam_odom

    if _slam_obj is None:
        _init_slam()
    if _slam_obj is False:
        return

    scan_mm = _preprocess_scan(points)
    now     = time.time()
    x_mm    = _x * 1000.0
    y_mm    = _y * 1000.0
    th_deg  = math.degrees(_th)

    if _last_slam_odom is not None:
        lx, ly, lth, lt = _last_slam_odom
        dx  = x_mm - lx
        dy  = y_mm - ly
        dxy = math.hypot(dx, dy) * math.copysign(1.0,
              dx * math.cos(math.radians(lth)) + dy * math.sin(math.radians(lth)))
        dth = th_deg - lth
        dt  = now - lt
        vel = (dxy, dth, dt)
    else:
        vel = None

    _last_slam_odom = (x_mm, y_mm, th_deg, now)

    with _slam_lock:
        try:
            # Call update with the API mode detected at init time
            if   _slam_update_mode == 'kw'  and vel is not None: _slam_obj.update(scan_mm, velocities=vel)
            elif _slam_update_mode == 'pos' and vel is not None: _slam_obj.update(scan_mm, vel)
            else:                                                 _slam_obj.update(scan_mm)

            sx, sy, sth = _slam_obj.getpos()
            _slam_pose   = [sx, sy, sth]
            mapbytes     = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
            _slam_obj.getmap(mapbytes)
            map_b64 = base64.b64encode(bytes(mapbytes)).decode()
            _map_queue.put_nowait({
                "type":   "slam",
                "map":    map_b64,
                "size":   MAP_SIZE_PIXELS,
                "meters": MAP_SIZE_METERS,
                "robot":  [round(sx, 1), round(sy, 1), round(sth, 2)],
            })
        except Exception as e:
            print(f"[slam] error: {e}")


# =============================================================================
# LiDAR reader thread — YDLiDAR SDK
# =============================================================================

def lidar_reader():
    global _lidar_status
    try:
        import ydlidar
    except ImportError:
        print("[lidar] ydlidar SDK not found — LiDAR disabled")
        return

    ydlidar.os_init()
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort,          LIDAR_PORT)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate,      LIDAR_BAUD)
    laser.setlidaropt(ydlidar.LidarPropLidarType,           ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType,          ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency,       LIDAR_FREQ)
    laser.setlidaropt(ydlidar.LidarPropSampleRate,          LIDAR_SAMPLE_RATE)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel,       LIDAR_SINGLE_CH)
    laser.setlidaropt(ydlidar.LidarPropReversion,           LIDAR_REVERSION)
    laser.setlidaropt(ydlidar.LidarPropInverted,            LIDAR_INVERTED)
    laser.setlidaropt(ydlidar.LidarPropMaxAngle,            180.0)
    laser.setlidaropt(ydlidar.LidarPropMinAngle,           -180.0)
    laser.setlidaropt(ydlidar.LidarPropMaxRange,            LIDAR_MAX_RANGE)
    laser.setlidaropt(ydlidar.LidarPropMinRange,            LIDAR_MIN_RANGE)
    laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
    laser.setlidaropt(ydlidar.LidarPropAutoReconnect,       True)

    if not laser.initialize():
        print(f"[lidar] Failed to initialize on {LIDAR_PORT}")
        return
    if not laser.turnOn():
        print("[lidar] Failed to start scanning")
        laser.disconnecting()
        return

    _lidar_status = "live"
    print(f"[lidar] ✓ YDLiDAR running on {LIDAR_PORT} @ {LIDAR_FREQ}Hz")
    scan = ydlidar.LaserScan()

    while ydlidar.os_isOk():
        r = laser.doProcessSimple(scan)
        if r and scan.points:
            points = []
            for p in scan.points:
                d = p.range
                if LIDAR_MIN_RANGE < d < LIDAR_MAX_RANGE:
                    points.append([round(math.degrees(p.angle), 2), round(d * 1000, 1)])
            if points:
                _lidar_queue.put_nowait(points)
                _slam_update(points)

    laser.turnOff()
    laser.disconnecting()
    _lidar_status = "offline"


# =============================================================================
# Broadcaster  — telemetry @20Hz, lidar @10Hz, SLAM map @1Hz
# =============================================================================

async def broadcaster(app):
    global _clients
    tick = 0
    while True:
        await asyncio.sleep(0.05)   # 20Hz base
        tick += 1

        # ── Telemetry ──────────────────────────────────────────────────────
        if _telemetry:
            msg = json.dumps({"type": "telemetry", **_telemetry,
                              "serial": _serial_status,
                              "lidar":  _lidar_status})
            async with _clients_lock:
                dead = set()
                for ws in _clients:
                    try:    await ws.send_str(msg)
                    except: dead.add(ws)
                _clients -= dead

        # ── LiDAR scan points ──────────────────────────────────────────────
        latest_scan = None
        while True:
            try:    latest_scan = _lidar_queue.get_nowait()
            except queue.Empty: break
        if latest_scan is not None:
            lmsg = json.dumps({"type": "lidar",
                               "points": latest_scan,
                               "rx": round(_x, 4),
                               "ry": round(_y, 4),
                               "rth": round(math.degrees(_th), 2),
                               "angle_offset": LIDAR_ANGLE_OFFSET_DEG})
            async with _clients_lock:
                dead = set()
                for ws in _clients:
                    try:    await ws.send_str(lmsg)
                    except: dead.add(ws)
                _clients -= dead

        # ── SLAM map (every ~2s = 40 ticks) ───────────────────────────────
        if tick % 40 == 0:
            latest_map = None
            while True:
                try:    latest_map = _map_queue.get_nowait()
                except queue.Empty: break
            if latest_map is not None:
                mmsg = json.dumps(latest_map)
                async with _clients_lock:
                    dead = set()
                    for ws in _clients:
                        try:    await ws.send_str(mmsg)
                        except: dead.add(ws)
                    _clients -= dead


# =============================================================================
# WebSocket handler
# =============================================================================

async def ws_handler(request):
    global _clients, _x, _y, _th, _vx
    global _prev_enc_l, _prev_enc_r, _prev_t, _last_slam_odom

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    async with _clients_lock:
        _clients.add(ws)
    print(f"[ws] {request.remote} connected  ({len(_clients)} clients)")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    d = json.loads(msg.data)
                    t = d.get("type")
                    if t == "cmd":
                        l = max(-255, min(255, int(d.get("left",  0))))
                        r = max(-255, min(255, int(d.get("right", 0))))
                        send_motor(l, r)
                    elif t == "stop":
                        send_motor(0, 0)
                    elif t == "reset_odom":
                        _x = _y = _th = _vx = 0.0
                        _prev_enc_l = _prev_enc_r = _prev_t = None
                        _last_slam_odom = None
                        print("[odom] Reset to origin")
                    elif t == "clear_map":
                        await ws.send_str(json.dumps({"type": "clear_map"}))
                        print("[map] Clear requested by client")
                except Exception as e:
                    print(f"[ws] msg error: {e}")
    finally:
        async with _clients_lock:
            _clients.discard(ws)
        print(f"[ws] {request.remote} disconnected")
    return ws


# =============================================================================
# HTML Dashboard — Apple/Google light design
# =============================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Speleo-X</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ── Design tokens ─────────────────────────────────────────────────────────── */
:root {
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'SF Mono', 'Menlo', 'Consolas', monospace;

  /* Colors */
  --bg:        #F2F2F7;
  --surface:   #FFFFFF;
  --text:      #1C1C1E;
  --text2:     #3A3A3C;
  --text3:     #6C6C70;
  --sep:       rgba(60,60,67,.12);
  --shadow:    0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
  --shadow-lg: 0 4px 16px rgba(0,0,0,.10);
  --blue:      #007AFF;
  --blue-d:    #0055B3;
  --green:     #34C759;
  --red:       #FF3B30;
  --orange:    #FF9500;
  --yellow:    #FFCC00;

  /* Map */
  --map-bg:   #EAECF0;
  --map-grid: rgba(0,0,0,.06);
  --map-free: #FFFFFF;
  --map-wall: #1C1C1E;
  --robot-c:  #007AFF;
  --path-c:   rgba(0,122,255,.18);
  --obs-c:    rgba(28,28,30,.82);

  --radius:   12px;
  --radius-sm: 8px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: var(--font); background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; overflow: hidden; }

/* ── Layout ─────────────────────────────────────────────────────────────────── */
#app {
  display: grid;
  grid-template-rows: 52px 1fr 136px;
  height: 100vh;
}

/* ── Header ─────────────────────────────────────────────────────────────────── */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--sep);
  z-index: 10;
}
.brand {
  font-size: 15px; font-weight: 600; letter-spacing: -.2px; color: var(--text);
  display: flex; align-items: center; gap: 8px;
}
.brand-icon {
  width: 24px; height: 24px; background: var(--blue); border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
}
.brand-icon svg { width: 14px; height: 14px; fill: #fff; }
.badges { display: flex; align-items: center; gap: 8px; }
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 500;
  background: var(--bg); color: var(--text3);
  border: 1px solid var(--sep);
  transition: all .2s;
}
.badge .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--text3); transition: background .3s;
}
.badge.live .dot { background: var(--green); box-shadow: 0 0 0 2px rgba(52,199,89,.2); }
.badge.live { color: var(--text2); }

/* ── Middle row ─────────────────────────────────────────────────────────────── */
#middle {
  display: grid;
  grid-template-columns: 260px 1fr;
  overflow: hidden;
}

/* ── Telemetry sidebar ───────────────────────────────────────────────────────── */
#sidebar {
  background: var(--surface);
  border-right: 1px solid var(--sep);
  padding: 16px 0;
  overflow-y: auto;
  display: flex; flex-direction: column; gap: 0;
}
.section { padding: 0 16px 16px; }
.section + .section { border-top: 1px solid var(--sep); padding-top: 14px; }
.section-title {
  font-size: 10px; font-weight: 600; letter-spacing: .8px;
  text-transform: uppercase; color: var(--text3);
  margin-bottom: 10px;
}
.trow {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 3px 0;
}
.trow .lbl { font-size: 12px; color: var(--text3); }
.trow .val {
  font-size: 13px; font-weight: 500; font-family: var(--font-mono);
  color: var(--text); letter-spacing: -.3px;
}
.trow .unit { font-size: 10px; color: var(--text3); margin-left: 2px; }

.sidebar-btn {
  display: flex; align-items: center; justify-content: center;
  width: 100%; height: 34px;
  border: 1px solid var(--sep); border-radius: var(--radius-sm);
  background: var(--bg); color: var(--text2);
  font-family: var(--font); font-size: 12px; font-weight: 500;
  cursor: pointer; transition: all .15s;
  margin-top: 4px;
}
.sidebar-btn:hover { background: #e8e8ed; }
.sidebar-btn:active { background: #dcdce1; transform: scale(.98); }
.sidebar-btn.danger { color: var(--red); border-color: rgba(255,59,48,.2); }
.sidebar-btn.danger:hover { background: rgba(255,59,48,.06); }

/* ── Map canvas ──────────────────────────────────────────────────────────────── */
#map-wrap {
  position: relative; background: var(--map-bg);
  overflow: hidden; cursor: grab;
}
#map-wrap:active { cursor: grabbing; }
#map-canvas { display: block; width: 100%; height: 100%; }

#map-overlay {
  position: absolute; top: 12px; right: 12px;
  display: flex; flex-direction: column; gap: 6px; z-index: 5;
}
.map-btn {
  width: 36px; height: 36px; border-radius: var(--radius-sm);
  background: var(--surface); border: 1px solid var(--sep);
  box-shadow: var(--shadow); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; color: var(--text2);
  transition: background .15s;
}
.map-btn:hover { background: #f0f0f5; }
.map-btn:active { background: #e0e0e8; transform: scale(.95); }

#map-scale-label {
  position: absolute; bottom: 10px; left: 12px;
  font-size: 10px; color: var(--text3); background: rgba(255,255,255,.8);
  padding: 3px 7px; border-radius: 6px;
  backdrop-filter: blur(4px);
}

/* ── Controls ────────────────────────────────────────────────────────────────── */
#controls {
  background: var(--surface);
  border-top: 1px solid var(--sep);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 28px; gap: 24px;
}

.dpad {
  display: grid;
  grid-template: repeat(3, 44px) / repeat(3, 44px);
  gap: 4px;
}
.dpad-btn {
  width: 44px; height: 44px;
  background: var(--bg); border: 1px solid var(--sep);
  border-radius: var(--radius-sm); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; color: var(--text2);
  transition: background .1s, transform .1s, box-shadow .1s;
  user-select: none; -webkit-user-select: none;
}
.dpad-btn:hover { background: #e8e8ed; }
.dpad-btn.pressed {
  background: var(--blue); color: #fff;
  border-color: var(--blue-d);
  box-shadow: 0 2px 8px rgba(0,122,255,.35);
  transform: scale(.94);
}
.dpad-center {
  background: rgba(255,59,48,.08); border-color: rgba(255,59,48,.25); color: var(--red);
}
.dpad-center.pressed { background: var(--red); color: #fff; border-color: var(--red); box-shadow: 0 2px 8px rgba(255,59,48,.35); }

.ctrl-middle { display: flex; flex-direction: column; align-items: center; gap: 14px; flex: 1; max-width: 240px; }
.speed-label { font-size: 11px; color: var(--text3); display: flex; justify-content: space-between; width: 100%; }
.speed-label strong { color: var(--text); font-weight: 600; }
input[type=range] {
  -webkit-appearance: none; appearance: none;
  width: 100%; height: 4px;
  background: linear-gradient(to right, var(--blue) 0%, var(--blue) var(--val, 55.9%), var(--sep) var(--val, 55.9%));
  border-radius: 2px; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
  background: var(--surface); border: 2px solid var(--blue);
  box-shadow: 0 1px 3px rgba(0,0,0,.2); cursor: pointer;
}

.stop-btn {
  width: 52px; height: 52px; border-radius: 50%;
  background: var(--red); border: none; cursor: pointer;
  color: #fff; font-size: 13px; font-weight: 600;
  box-shadow: 0 2px 8px rgba(255,59,48,.35);
  transition: transform .12s, box-shadow .12s;
}
.stop-btn:active { transform: scale(.9); box-shadow: 0 1px 4px rgba(255,59,48,.4); }

.ctrl-right { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.kbd-hint { font-size: 10px; color: var(--text3); text-align: center; line-height: 1.5; }
.kbd-hint kbd {
  display: inline-block; padding: 1px 5px; border-radius: 4px;
  background: var(--bg); border: 1px solid var(--sep);
  font-family: var(--font-mono); font-size: 10px; color: var(--text2);
}
</style>
</head>
<body>
<div id="app">

  <!-- ── Header ─────────────────────────────────────────────────────── -->
  <header>
    <div class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 16 16"><path d="M8 1L14 4v6l-6 4-6-4V4z"/></svg>
      </div>
      Speleo-X
    </div>
    <div class="badges">
      <div class="badge" id="badge-ws"><span class="dot"></span>WebSocket</div>
      <div class="badge" id="badge-serial"><span class="dot"></span>Serial</div>
      <div class="badge" id="badge-lidar"><span class="dot"></span>LiDAR</div>
      <div class="badge" id="badge-slam"><span class="dot"></span>SLAM</div>
    </div>
  </header>

  <!-- ── Middle: sidebar + map ─────────────────────────────────────── -->
  <div id="middle">

    <!-- Telemetry sidebar -->
    <div id="sidebar">

      <div class="section">
        <div class="section-title">Position</div>
        <div class="trow"><span class="lbl">X</span><span><span class="val" id="t-x">0.000</span><span class="unit">m</span></span></div>
        <div class="trow"><span class="lbl">Y</span><span><span class="val" id="t-y">0.000</span><span class="unit">m</span></span></div>
        <div class="trow"><span class="lbl">Heading</span><span><span class="val" id="t-th">0.00</span><span class="unit">°</span></span></div>
        <div class="trow"><span class="lbl">Speed</span><span><span class="val" id="t-vx">0.000</span><span class="unit">m/s</span></span></div>
      </div>

      <div class="section">
        <div class="section-title">Accelerometer</div>
        <div class="trow"><span class="lbl">X</span><span><span class="val" id="t-ax">0.00</span><span class="unit">m/s²</span></span></div>
        <div class="trow"><span class="lbl">Y</span><span><span class="val" id="t-ay">0.00</span><span class="unit">m/s²</span></span></div>
        <div class="trow"><span class="lbl">Z</span><span><span class="val" id="t-az">0.00</span><span class="unit">m/s²</span></span></div>
      </div>

      <div class="section">
        <div class="section-title">Gyroscope</div>
        <div class="trow"><span class="lbl">Yaw rate</span><span><span class="val" id="t-gz">0.0000</span><span class="unit">rad/s</span></span></div>
      </div>

      <div class="section">
        <div class="section-title">Encoders</div>
        <div class="trow"><span class="lbl">Left</span><span class="val" id="t-encl">—</span></div>
        <div class="trow"><span class="lbl">Right</span><span class="val" id="t-encr">—</span></div>
      </div>

      <div class="section">
        <div class="section-title">Actions</div>
        <button class="sidebar-btn" id="btn-reset-odom">Reset Odometry</button>
        <button class="sidebar-btn" id="btn-clear-map">Clear Map</button>
        <button class="sidebar-btn danger" id="btn-estop">Emergency Stop</button>
      </div>

    </div>

    <!-- Map canvas -->
    <div id="map-wrap">
      <canvas id="map-canvas"></canvas>
      <div id="map-overlay">
        <button class="map-btn" id="map-zoom-in"  title="Zoom in">+</button>
        <button class="map-btn" id="map-zoom-out" title="Zoom out">−</button>
        <button class="map-btn" id="map-center"   title="Center on robot">⊙</button>
      </div>
      <div id="map-scale-label" id="map-scale-lbl">— m/div</div>
    </div>

  </div>

  <!-- ── Controls bar ───────────────────────────────────────────────── -->
  <div id="controls">

    <!-- D-Pad -->
    <div class="dpad">
      <div></div>
      <button class="dpad-btn" id="btn-w" data-key="w">▲</button>
      <div></div>
      <button class="dpad-btn" id="btn-a" data-key="a">◀</button>
      <button class="dpad-btn dpad-center" id="btn-spc" data-key=" ">■</button>
      <button class="dpad-btn" id="btn-d" data-key="d">▶</button>
      <div></div>
      <button class="dpad-btn" id="btn-s" data-key="s">▼</button>
      <div></div>
    </div>

    <!-- Speed + labels -->
    <div class="ctrl-middle">
      <div class="speed-label">
        <span>Speed</span>
        <strong id="speed-val">200</strong>
      </div>
      <input type="range" id="speed-slider" min="50" max="255" value="200" step="5">
    </div>

    <!-- Stop -->
    <button class="stop-btn" id="btn-stop">STOP</button>

    <!-- Keyboard hint -->
    <div class="ctrl-right">
      <div class="kbd-hint">
        <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to drive<br>
        <kbd>Space</kbd> to stop
      </div>
    </div>

  </div>
</div>

<script>
// ── Constants ─────────────────────────────────────────────────────────────────
const GRID_RES     = 0.05;    // 5 cm obstacle grid
const MAP_SCALE_PX = 60;      // canvas pixels per metre (default)
const PATH_MAX     = 5000;    // max stored path points

// ── State ─────────────────────────────────────────────────────────────────────
let robotX = 0, robotY = 0, robotTh = 0;  // world metres / degrees
let pathPts   = [];            // [{x,y}] world metres
let obstacles = new Map();     // "gx,gy" → count
let slamMap   = null;          // Uint8Array of slam map
let slamMeta  = null;          // {size, meters, robot:[x_mm,y_mm,th]}

let mapScale  = MAP_SCALE_PX; // current zoom
let panX = 0, panY = 0;       // manual pan offset from robot-centric center
let slamAvail = false;

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws, reconnTimer;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen  = () => { setWsBadge(true); clearTimeout(reconnTimer); };
  ws.onclose = ws.onerror = () => { setWsBadge(false); reconnTimer = setTimeout(connect, 2000); };
  ws.onmessage = e => handle(JSON.parse(e.data));
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
connect();

// ── Message dispatch ──────────────────────────────────────────────────────────
function handle(d) {
  if (d.type === 'telemetry') onTelemetry(d);
  if (d.type === 'lidar')     onLidar(d);
  if (d.type === 'slam')      onSlam(d);
  if (d.type === 'clear_map') clearMapData();
}

function onTelemetry(d) {
  robotX  = d.x;
  robotY  = d.y;
  robotTh = d.th;
  setText('t-x',    d.x.toFixed(3));
  setText('t-y',    d.y.toFixed(3));
  setText('t-th',   d.th.toFixed(2));
  setText('t-vx',   d.vx.toFixed(3));
  setText('t-ax',   d.ax.toFixed(2));
  setText('t-ay',   d.ay.toFixed(2));
  setText('t-az',   d.az.toFixed(2));
  setText('t-gz',   d.gz.toFixed(4));
  setText('t-encl', d.enc_l);
  setText('t-encr', d.enc_r);
  setBadge('badge-serial', d.serial === 'live');
  setBadge('badge-lidar',  d.lidar  === 'live');

  // Record path
  if (pathPts.length === 0 ||
      Math.hypot(robotX - pathPts[pathPts.length-1].x,
                 robotY - pathPts[pathPts.length-1].y) > 0.02) {
    pathPts.push({x: robotX, y: robotY});
    if (pathPts.length > PATH_MAX) pathPts.shift();
  }
}

function onLidar(d) {
  // d.points = [[angle_deg, dist_mm], ...]
  // d.rx, d.ry = robot world position at scan time (metres)
  // d.rth      = robot heading (degrees) at scan time
  // d.angle_offset = LiDAR-to-robot frame rotation (degrees)
  const rx   = d.rx,  ry  = d.ry;
  const rthR = d.rth * Math.PI / 180;
  const cosR = Math.cos(rthR), sinR = Math.sin(rthR);
  const offR = (d.angle_offset || 0) * Math.PI / 180;

  for (const [angleDeg, distMm] of d.points) {
    const distM  = distMm / 1000;
    const aR     = angleDeg * Math.PI / 180 + offR;  // apply lidar frame offset
    // Local lidar frame → world frame (rotate by robot heading)
    const lx = distM * Math.cos(aR);
    const ly = distM * Math.sin(aR);
    const wx = rx + cosR * lx - sinR * ly;
    const wy = ry + sinR * lx + cosR * ly;
    // Bin to grid
    const gx  = Math.round(wx / GRID_RES);
    const gy  = Math.round(wy / GRID_RES);
    const key = `${gx},${gy}`;
    obstacles.set(key, Math.min(20, (obstacles.get(key) || 0) + 1));
  }
}

function onSlam(d) {
  // d.map = base64 byte string, d.size, d.meters, d.robot=[x_mm,y_mm,th]
  const raw = atob(d.map);
  slamMap  = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) slamMap[i] = raw.charCodeAt(i);
  slamMeta = d;
  slamAvail = true;
  setBadge('badge-slam', true);
}

function clearMapData() {
  obstacles.clear();
  pathPts = [];
  slamMap = null; slamMeta = null; slamAvail = false;
  setBadge('badge-slam', false);
}

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas  = document.getElementById('map-canvas');
const ctx     = canvas.getContext('2d');
const offC    = document.createElement('canvas');
const offCtx  = offC.getContext('2d');

function resizeCanvas() {
  const wrap = document.getElementById('map-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ── World → canvas transform (robot-centred) ──────────────────────────────────
function w2c(wx, wy) {
  const cx = canvas.width / 2 + panX;
  const cy = canvas.height / 2 + panY;
  return [cx + (wx - robotX) * mapScale, cy - (wy - robotY) * mapScale];
}

// ── Render loop ───────────────────────────────────────────────────────────────
function render() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // ── Background grid ────────────────────────────────────────────────────────
  ctx.fillStyle = '#EAECF0';
  ctx.fillRect(0, 0, W, H);
  drawGrid();

  // ── SLAM map (if available) ────────────────────────────────────────────────
  if (slamMap && slamMeta) drawSlamMap();

  // ── Obstacle points ────────────────────────────────────────────────────────
  drawObstacles();

  // ── Path trail ─────────────────────────────────────────────────────────────
  if (pathPts.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0,122,255,0.20)';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';
    pathPts.forEach((p, i) => {
      const [cx, cy] = w2c(p.x, p.y);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.stroke();
  }

  // ── Robot ──────────────────────────────────────────────────────────────────
  drawRobot();

  // ── Scale label ────────────────────────────────────────────────────────────
  const scaleDist = (1 / mapScale).toFixed(2);
  document.getElementById('map-scale-label').textContent = `${scaleDist} m/px  ·  ${Math.round(mapScale)} px/m`;

  requestAnimationFrame(render);
}
render();

function drawGrid() {
  const step   = mapScale; // 1m grid lines
  const cx     = canvas.width / 2 + panX;
  const cy     = canvas.height / 2 + panY;
  const offX   = ((cx % step) + step) % step;
  const offY   = ((cy % step) + step) % step;
  ctx.strokeStyle = 'rgba(0,0,0,.05)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  for (let x = offX; x < canvas.width;  x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); }
  for (let y = offY; y < canvas.height; y += step) { ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);  }
  ctx.stroke();
  // Origin crosshair
  const [ox, oy] = w2c(0, 0);
  ctx.strokeStyle = 'rgba(0,0,0,.15)';
  ctx.lineWidth   = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, canvas.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(canvas.width, oy);  ctx.stroke();
  ctx.setLineDash([]);
}

function drawSlamMap() {
  const { size, meters, robot: [rx_mm, ry_mm] } = slamMeta;

  // Draw to offscreen canvas
  offC.width = offC.height = size;
  const img = offCtx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = slamMap[i];
    // breezyslam: 0=unknown, up to 127=free space
    let r, g, b;
    if (v === 0) { r = g = b = 210; }            // unknown — mid gray
    else { const t = v/127; r=g=b=Math.round(210+t*45); }  // free — white
    img.data[i*4]=r; img.data[i*4+1]=g; img.data[i*4+2]=b; img.data[i*4+3]=255;
  }
  offCtx.putImageData(img, 0, 0);

  // SLAM robot pos → map pixel
  // getpos() returns ABSOLUTE mm from map bottom-left corner.
  // Robot starts at map center = (MAP_SIZE_METERS/2*1000, MAP_SIZE_METERS/2*1000) mm
  const mmPerPx = meters * 1000 / size;
  const slamPx  = rx_mm / mmPerPx;          // absolute map pixel X
  const slamPy  = size - ry_mm / mmPerPx;   // Y flipped (screen Y goes down)

  // Scale: canvas pixels per map pixel
  const s = mapScale * meters / size;

  // Draw map so that slamPx/slamPy lands exactly on the robot canvas position
  const [robCx, robCy] = w2c(robotX, robotY);
  const drawX = robCx - slamPx * s;
  const drawY = robCy - slamPy * s;

  ctx.save();
  ctx.globalAlpha = 0.70;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offC, drawX, drawY, size * s, size * s);
  ctx.restore();
}

function drawObstacles() {
  const pxSize = Math.max(2, GRID_RES * mapScale * 0.9);
  ctx.fillStyle = 'rgba(28,28,30,0.75)';
  for (const key of obstacles.keys()) {
    const [gx, gy] = key.split(',').map(Number);
    const wx = gx * GRID_RES, wy = gy * GRID_RES;
    const [cx, cy] = w2c(wx, wy);
    if (cx < -4 || cx > canvas.width + 4 || cy < -4 || cy > canvas.height + 4) continue;
    ctx.fillRect(cx - pxSize/2, cy - pxSize/2, pxSize, pxSize);
  }
}

function drawRobot() {
  const [cx, cy] = w2c(robotX, robotY);
  const angle = (robotTh - 90) * Math.PI / 180;  // screen: 0° = up
  const sz = 14;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(angle);

  // Shadow
  ctx.shadowColor = 'rgba(0,122,255,0.3)';
  ctx.shadowBlur = 10;

  // Body
  ctx.beginPath();
  ctx.moveTo(0, -sz);
  ctx.lineTo(sz * 0.55,  sz * 0.65);
  ctx.lineTo(0,          sz * 0.25);
  ctx.lineTo(-sz * 0.55, sz * 0.65);
  ctx.closePath();
  ctx.fillStyle = '#007AFF';
  ctx.fill();

  // White center dot
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.arc(0, 0, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();

  ctx.restore();
}

// ── Map controls ──────────────────────────────────────────────────────────────
document.getElementById('map-zoom-in').onclick  = () => { mapScale = Math.min(200, mapScale * 1.3); };
document.getElementById('map-zoom-out').onclick = () => { mapScale = Math.max(10,  mapScale / 1.3); };
document.getElementById('map-center').onclick   = () => { panX = 0; panY = 0; };

// Scroll to zoom
document.getElementById('map-wrap').addEventListener('wheel', e => {
  e.preventDefault();
  mapScale = e.deltaY < 0
    ? Math.min(200, mapScale * 1.1)
    : Math.max(10,  mapScale / 1.1);
}, { passive: false });

// Drag to pan
let dragging = false, dragX = 0, dragY = 0;
canvas.addEventListener('mousedown', e => { dragging = true; dragX = e.clientX; dragY = e.clientY; });
canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  panX += e.clientX - dragX; panY += e.clientY - dragY;
  dragX = e.clientX; dragY = e.clientY;
});
window.addEventListener('mouseup', () => { dragging = false; });

// ── Telemetry helpers ─────────────────────────────────────────────────────────
function setText(id, v) { document.getElementById(id).textContent = v; }
function setBadge(id, live) {
  const el = document.getElementById(id);
  el.classList.toggle('live', live);
}
function setWsBadge(live) { setBadge('badge-ws', live); }

// ── Sidebar actions ───────────────────────────────────────────────────────────
document.getElementById('btn-reset-odom').onclick = () => send({ type: 'reset_odom' });
document.getElementById('btn-clear-map').onclick  = () => { clearMapData(); send({ type: 'clear_map' }); };
document.getElementById('btn-estop').onclick      = () => send({ type: 'stop' });

// ── Speed slider ──────────────────────────────────────────────────────────────
const speedSlider = document.getElementById('speed-slider');
function updateSliderCSS() {
  const pct = ((speedSlider.value - speedSlider.min) / (speedSlider.max - speedSlider.min) * 100).toFixed(1) + '%';
  speedSlider.style.setProperty('--val', pct);
  document.getElementById('speed-val').textContent = speedSlider.value;
}
speedSlider.addEventListener('input', updateSliderCSS);
updateSliderCSS();

const speed = () => parseInt(speedSlider.value);

// ── Motor command helpers ──────────────────────────────────────────────────────
function sendCmd(l, r) { send({ type: 'cmd', left: l, right: r }); }
function sendStop()    { send({ type: 'stop' }); }

// ── Key → command map ─────────────────────────────────────────────────────────
const KEY_CMD = {
  w:   () => sendCmd( speed(),  speed()),
  s:   () => sendCmd(-speed(), -speed()),
  a:   () => sendCmd(-speed(),  speed()),
  d:   () => sendCmd( speed(), -speed()),
  ' ': () => sendStop(),
};
const DRIVE_KEYS = new Set(['w','s','a','d']);

// ── Continuous hold (interval repeats every 250ms) ────────────────────────────
let holdKey = null, holdTimer = null;

function startHold(key) {
  if (!KEY_CMD[key]) return;
  if (holdKey === key) return;
  stopHold();
  holdKey = key;
  KEY_CMD[key]();
  if (DRIVE_KEYS.has(key))
    holdTimer = setInterval(() => KEY_CMD[holdKey] && KEY_CMD[holdKey](), 250);
  highlightKey(key, true);
}

function stopHold(key) {
  if (key && key !== holdKey) return;
  clearInterval(holdTimer); holdTimer = null;
  if (holdKey && DRIVE_KEYS.has(holdKey)) sendStop();
  highlightKey(holdKey, false);
  holdKey = null;
}

const heldKeys = new Set();
document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (e.repeat) return;
  if (KEY_CMD[k]) { heldKeys.add(k); startHold(k); }
  if (k === ' ')  e.preventDefault();
});
document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  heldKeys.delete(k);
  stopHold(k);
});

function highlightKey(key, on) {
  document.querySelectorAll('.dpad-btn[data-key]').forEach(b => {
    if (b.dataset.key === key) b.classList.toggle('pressed', on);
  });
}

// ── On-screen D-pad buttons ───────────────────────────────────────────────────
document.querySelectorAll('.dpad-btn[data-key]').forEach(btn => {
  const k = btn.dataset.key;
  btn.addEventListener('pointerdown', e => { e.preventDefault(); btn.setPointerCapture(e.pointerId); startHold(k); });
  btn.addEventListener('pointerup',   () => stopHold(k));
  btn.addEventListener('pointerout',  () => stopHold(k));
});
document.getElementById('btn-stop').addEventListener('pointerdown', () => sendStop());
</script>
</body>
</html>"""


# =============================================================================
# aiohttp routes
# =============================================================================

async def index(request):
    return web.Response(text=HTML, content_type="text/html")


# =============================================================================
# App startup
# =============================================================================

async def on_startup(app):
    global _ser, _clients_lock
    _clients_lock = asyncio.Lock()
    _ser = open_serial()
    threading.Thread(target=serial_reader, daemon=True).start()
    threading.Thread(target=watchdog,      daemon=True).start()
    threading.Thread(target=lidar_reader,  daemon=True).start()
    asyncio.create_task(broadcaster(app))
    print(f"\n{'='*54}")
    print(f"  Speleo-X Dashboard  →  http://<PI_IP>:{PORT}")
    print(f"  STM32 serial : {', '.join(SERIAL_PORTS)}")
    print(f"  LiDAR port   : {LIDAR_PORT}")
    print(f"  SLAM map     : {MAP_SIZE_PIXELS}×{MAP_SIZE_PIXELS} px  /  {MAP_SIZE_METERS}m")
    print(f"{'='*54}\n")


app = web.Application()
app.router.add_get("/",   index)
app.router.add_get("/ws", ws_handler)
app.on_startup.append(on_startup)

if __name__ == "__main__":
    web.run_app(app, host=HOST, port=PORT)
