#!/usr/bin/env python3
"""
speleo_foxglove.py  —  Speleo-X Pure Python ↔ Foxglove Bridge
==============================================================

No ROS required. Streams STM32 telemetry directly to Foxglove Studio
over a WebSocket on port 8765.

Install:
    pip3 install foxglove-websocket pyserial

Run:
    python3 speleo_foxglove.py

Then in Foxglove Studio:
    Open Connection → Foxglove WebSocket → ws://<PI_IP>:8765

Channels published:
    /imu       — sensor_msgs/Imu        (accelerometer + gyro)
    /odom      — nav_msgs/Odometry      (dead-reckoning position)
    /enc_ticks — foxglove.RawImage      (raw tick debug string)
    /motors    — current motor PWM state

Motor commands received from Foxglove:
    /cmd_vel   — geometry_msgs/Twist    (linear.x, angular.z)
"""

import asyncio
import json
import math
import queue
import serial
import threading
import time
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId

# ── Configuration — edit these ────────────────────────────────────────────────
SERIAL_PORT     = '/dev/bluepill'   # or /dev/ttyUSB0
BAUD_RATE       = 115200

WHEEL_RADIUS    = 0.035             # metres
WHEEL_BASE      = 0.20             # metres (calibrate with spin test)
TICKS_PER_REV_L = 725              # measured left wheel
TICKS_PER_REV_R = 711              # measured right wheel

MAX_LINEAR_MPS  = 0.3              # m/s at PWM=255
MAX_ANGULAR_RPS = 2.0              # rad/s at PWM=255

ACCEL_SCALE     = 16384.0          # LSB/g  (MPU6050 ±2g)
GYRO_SCALE      = 131.0            # LSB/(°/s) (MPU6050 ±250°/s)
GRAVITY         = 9.81             # m/s²
# ─────────────────────────────────────────────────────────────────────────────


def pack_time(t: float) -> dict:
    """Convert float seconds to {sec, nsec} dict for Foxglove schemas."""
    sec  = int(t)
    nsec = int((t - sec) * 1e9)
    return {"sec": sec, "nsec": nsec}


class SpeleoFoxglove(FoxgloveServerListener):

    def __init__(self):
        # ── Odometry state ────────────────────────────────────────────────────
        self._x   = 0.0
        self._y   = 0.0
        self._th  = 0.0
        self._vx  = 0.0
        self._wz  = 0.0
        self._prev_enc_l = None
        self._prev_enc_r = None
        self._prev_t     = None

        # ── Motor state ───────────────────────────────────────────────────────
        self._left_pwm  = 0
        self._right_pwm = 0
        self._last_cmd  = time.time()
        self._lock      = threading.Lock()

        # ── Serial ───────────────────────────────────────────────────────────
        import os
        ports_to_try = [SERIAL_PORT, '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0']
        
        for port in ports_to_try:
            if os.path.exists(port):
                try:
                    self.ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
                    print(f"[serial] Opened {port} @ {BAUD_RATE}")
                    break
                except serial.SerialException as e:
                    print(f"[serial] Found {port} but got permission error: {e}")
                    print(f"         Try running: sudo usermod -aG dialout $USER")
                    print(f"         Then LOG OUT and LOG BACK IN.")
                    raise
        else:
            raise FileNotFoundError(f"Could not find any serial ports! Checked: {ports_to_try}")

        # ── Foxglove server & channels (set in run()) ─────────────────────────
        self._server   = None
        self._ch_imu   = None
        self._ch_odom  = None
        self._ch_enc   = None
        self._ch_motor = None
        # Thread-safe queue: serial thread writes, async sender reads
        self._send_queue = queue.SimpleQueue()

    # =========================================================================
    # FoxgloveServerListener — called when Foxglove publishes back to us
    # =========================================================================
    def on_client_advertise(self, server, client_id, channel):
        """Called when Foxglove advertises a channel it wants to publish on."""
        print(f"[foxglove] Client advertised: topic='{channel.get('topic')}' "
              f"schema='{channel.get('schemaName')}'")

    def on_message_data(self, server, client_id, channel_id, data: bytes):
        """Receive /cmd_vel from Foxglove and drive motors."""
        print(f"[cmd_vel] raw data: {data[:120]}")   # ← debug: remove once working
        try:
            msg = json.loads(data.decode())
            lx = msg.get("linear",  {}).get("x", 0.0)
            wz = msg.get("angular", {}).get("z", 0.0)

            left_pwm  = int(lx / MAX_LINEAR_MPS  * 255
                           - wz / MAX_ANGULAR_RPS * 255)
            right_pwm = int(lx / MAX_LINEAR_MPS  * 255
                           + wz / MAX_ANGULAR_RPS * 255)

            left_pwm  = max(-255, min(255, left_pwm))
            right_pwm = max(-255, min(255, right_pwm))

            self._send_motor(left_pwm, right_pwm)
            self._last_cmd = time.time()
            print(f"[cmd_vel] L={left_pwm} R={right_pwm}")
        except Exception as e:
            print(f"[cmd_vel] parse error: {e}  raw={data[:80]}")

    # =========================================================================
    # Parse one STM32 telemetry line
    # =========================================================================
    def _parse_line(self, line: str):
        if not line or line.startswith('#'):
            return

        try:
            fields = {}
            for token in line.split():
                if '=' in token:
                    k, v = token.split('=', 1)
                    fields[k] = int(v)

            required = {'ENC_L', 'ENC_R', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'}
            if not required.issubset(fields.keys()):
                return

            enc_l = fields['ENC_L']
            enc_r = fields['ENC_R']

            # SI unit conversion
            ax = fields['AX'] / ACCEL_SCALE * GRAVITY
            ay = fields['AY'] / ACCEL_SCALE * GRAVITY
            az = fields['AZ'] / ACCEL_SCALE * GRAVITY
            gx = math.radians(fields['GX'] / GYRO_SCALE)
            gy = math.radians(fields['GY'] / GYRO_SCALE)
            gz = math.radians(fields['GZ'] / GYRO_SCALE)

            now = time.time()
            stamp = pack_time(now)

            # ── Dead-reckoning odometry ───────────────────────────────────────
            if self._prev_enc_l is not None and self._prev_t is not None:
                d_enc_l = enc_l - self._prev_enc_l
                d_enc_r = enc_r - self._prev_enc_r
                dt      = now - self._prev_t

                d_left   = (d_enc_l / TICKS_PER_REV_L) * 2 * math.pi * WHEEL_RADIUS
                d_right  = (d_enc_r / TICKS_PER_REV_R) * 2 * math.pi * WHEEL_RADIUS
                d_center = (d_left + d_right) / 2.0
                d_theta  = (d_right - d_left) / WHEEL_BASE

                self._th += d_theta
                self._th  = math.atan2(math.sin(self._th), math.cos(self._th))
                self._x  += d_center * math.cos(self._th)
                self._y  += d_center * math.sin(self._th)
                self._vx  = d_center / dt if dt > 0 else 0.0
                self._wz  = gz

            self._prev_enc_l = enc_l
            self._prev_enc_r = enc_r
            self._prev_t     = now

            now_ns = int(now * 1e9)  # Foxglove needs nanoseconds
            qz = math.sin(self._th / 2.0)
            qw = math.cos(self._th / 2.0)

            # ── Queue all messages for the async sender task ───────────────────
            if self._send_queue and self._ch_imu:
                imu_msg = json.dumps({
                    "timestamp": stamp,
                    "frame_id":  "base_link",
                    "orientation": {"x": 0.0, "y": 0.0, "z": qz, "w": qw},
                    "angular_velocity":    {"x": gx, "y": gy, "z": gz},
                    "linear_acceleration": {"x": ax, "y": ay, "z": az},
                }).encode()
                self._send_queue.put_nowait((self._ch_imu, now_ns, imu_msg))

            if self._send_queue and self._ch_odom:
                odom_msg = json.dumps({
                    "timestamp":      stamp,
                    "frame_id":       "odom",
                    "child_frame_id": "base_link",
                    "pose": {
                        "pose": {
                            "position":    {"x": self._x, "y": self._y, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": qz, "w": qw},
                        },
                        "covariance": [0.01,0,0,0,0,0, 0,0.01,0,0,0,0,
                                       0,0,0,0,0,0, 0,0,0,0,0,0,
                                       0,0,0,0,0,0, 0,0,0,0,0,0.05]
                    },
                    "twist": {
                        "twist": {
                            "linear":  {"x": self._vx, "y": 0.0, "z": 0.0},
                            "angular": {"x": 0.0, "y": 0.0, "z": self._wz},
                        },
                        "covariance": [0.01,0,0,0,0,0, 0,0.01,0,0,0,0,
                                       0,0,0,0,0,0, 0,0,0,0,0,0,
                                       0,0,0,0,0,0, 0,0,0,0,0,0.05]
                    }
                }).encode()
                self._send_queue.put_nowait((self._ch_odom, now_ns, odom_msg))

            if self._send_queue and self._ch_enc:
                enc_msg = json.dumps({
                    "timestamp": stamp,
                    "data": f"ENC_L={enc_l}  ENC_R={enc_r}"
                }).encode()
                self._send_queue.put_nowait((self._ch_enc, now_ns, enc_msg))

            if self._send_queue and self._ch_motor:
                motor_msg = json.dumps({
                    "timestamp": stamp,
                    "data": f"LEFT_PWM={self._left_pwm}  RIGHT_PWM={self._right_pwm}"
                }).encode()
                self._send_queue.put_nowait((self._ch_motor, now_ns, motor_msg))

        except Exception as e:
            print(f"[parse] error on '{line}': {e}")

    # =========================================================================
    # Serial reader thread — manual byte buffer to prevent line merging
    # =========================================================================
    def _serial_thread(self):
        buf = b''
        while True:
            try:
                # Read however many bytes are waiting (at least 1 to block)
                chunk = self.ser.read(max(1, self.ser.in_waiting))
                if chunk:
                    buf += chunk
                    # Process every complete line in the buffer
                    while b'\n' in buf:
                        line_bytes, buf = buf.split(b'\n', 1)
                        text = line_bytes.decode('utf-8', errors='ignore').strip()
                        if text:
                            self._parse_line(text)
            except Exception as e:
                print(f"[serial] read error: {e}")
                time.sleep(0.1)

    # =========================================================================
    # Watchdog thread — stops motors if no cmd_vel for >1 s
    # =========================================================================
    def _watchdog_thread(self):
        while True:
            time.sleep(0.2)
            if (time.time() - self._last_cmd > 1.0 and
                    (self._left_pwm != 0 or self._right_pwm != 0)):
                print("[watchdog] No cmd_vel — stopping motors")
                self._send_motor(0, 0)

    # =========================================================================
    # Send motor command over serial
    # =========================================================================
    def _send_motor(self, left: int, right: int):
        self._left_pwm  = left
        self._right_pwm = right
        cmd = f'm,{left},{right}\n'.encode()
        with self._lock:
            try:
                self.ser.write(cmd)
            except Exception as e:
                print(f"[serial] write error: {e}")

    # =========================================================================
    # Main async run loop
    # =========================================================================
    async def run(self):
        # _send_queue is already created in __init__ as a thread-safe SimpleQueue

        async with FoxgloveServer(
            host="0.0.0.0",
            port=8765,
            name="Speleo-X Bridge",
        ) as server:
            server.set_listener(self)
            self._server = server

            # ── Register channels ─────────────────────────────────────────────
            self._ch_imu = await server.add_channel({
                "topic":    "/imu",
                "encoding": "json",
                "schemaName": "ros.sensor_msgs.Imu",
                "schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "timestamp":           {"type": "object"},
                        "frame_id":            {"type": "string"},
                        "orientation":         {"type": "object"},
                        "angular_velocity":    {"type": "object"},
                        "linear_acceleration": {"type": "object"},
                    }
                })
            })

            self._ch_odom = await server.add_channel({
                "topic":    "/odom",
                "encoding": "json",
                "schemaName": "ros.nav_msgs.Odometry",
                "schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "timestamp":      {"type": "object"},
                        "frame_id":       {"type": "string"},
                        "child_frame_id": {"type": "string"},
                        "pose":           {"type": "object"},
                        "twist":          {"type": "object"},
                    }
                })
            })

            self._ch_enc = await server.add_channel({
                "topic":    "/enc_ticks",
                "encoding": "json",
                "schemaName": "foxglove.Log",
                "schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "object"},
                        "data":      {"type": "string"},
                    }
                })
            })

            self._ch_motor = await server.add_channel({
                "topic":    "/motor_state",
                "encoding": "json",
                "schemaName": "foxglove.Log",
                "schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "object"},
                        "data":      {"type": "string"},
                    }
                })
            })

            # ── Start background threads ──────────────────────────────────────
            threading.Thread(target=self._serial_thread,   daemon=True).start()
            threading.Thread(target=self._watchdog_thread, daemon=True).start()

            print("[foxglove] Server running on ws://0.0.0.0:8765")
            print("[foxglove] Open Foxglove Studio → Open Connection → Foxglove WebSocket")
            print(f"[foxglove] URL: ws://<YOUR_PI_IP>:8765")
            print()
            print("Channels:")
            print("  /imu          — accelerometer + gyroscope (SI units)")
            print("  /odom         — dead-reckoning odometry")
            print("  /enc_ticks    — raw encoder tick counts")
            print("  /motor_state  — current PWM values")
            print()
            print("Press Ctrl+C to stop.")

            # ── Async sender: polls thread-safe queue and sends to Foxglove ──
            async def _sender():
                while True:
                    try:
                        # Drain all pending messages without blocking
                        while True:
                            try:
                                ch, ts_ns, data = self._send_queue.get_nowait()
                                await server.send_message(ch, ts_ns, data)
                            except queue.Empty:
                                break
                    except Exception as e:
                        print(f"[sender] error: {e}")
                    await asyncio.sleep(0.005)  # poll at 200Hz

            asyncio.create_task(_sender())

            # Keep running forever
            while True:
                await asyncio.sleep(1.0)


def main():
    bridge = SpeleoFoxglove()
    run_cancellable(bridge.run())


if __name__ == "__main__":
    main()
