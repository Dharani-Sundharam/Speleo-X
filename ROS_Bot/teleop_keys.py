#!/usr/bin/env python3
"""
teleop_keys.py — Simple keyboard control for Speleo-X
======================================================
Run this on the Raspberry Pi in a second terminal while
speleo_foxglove.py is running (or standalone).

Controls:
  W / ↑   — Forward
  S / ↓   — Reverse
  A / ←   — Turn Left
  D / →   — Turn Right
  SPACE   — Stop
  Q       — Quit

Usage:
  python3 teleop_keys.py
  python3 teleop_keys.py --port /dev/ttyUSB0  (if bluepill symlink not set)
"""

import sys
import tty
import termios
import serial
import time
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_PORTS = ['/dev/bluepill', '/dev/ttyUSB0', '/dev/ttyUSB1']
BAUD_RATE     = 115200
DRIVE_PWM     = 200   # -255 to 255
TURN_PWM      = 180
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = {
    'w': ( DRIVE_PWM,  DRIVE_PWM,  "FORWARD"),
    's': (-DRIVE_PWM, -DRIVE_PWM,  "REVERSE"),
    'a': (-TURN_PWM,   TURN_PWM,   "TURN LEFT"),
    'd': ( TURN_PWM,  -TURN_PWM,   "TURN RIGHT"),
    ' ': (0,           0,           "STOP"),
}

def get_key():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle arrow keys (they send 3-byte escape sequences)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            arrow = {'\x41': 'w', '\x42': 's', '\x43': 'd', '\x44': 'a'}
            ch = arrow.get(ch3, '')
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def open_serial(port_arg):
    import os
    ports = [port_arg] if port_arg else DEFAULT_PORTS
    for p in ports:
        if os.path.exists(p):
            try:
                s = serial.Serial(p, BAUD_RATE, timeout=0.1)
                print(f"Opened {p}")
                return s
            except serial.SerialException as e:
                print(f"Cannot open {p}: {e}")
    raise RuntimeError(f"No serial port found. Tried: {ports}")


def send(ser, left, right):
    ser.write(f'm,{left},{right}\n'.encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=None, help='Serial port override')
    args = parser.parse_args()

    ser = open_serial(args.port)
    time.sleep(0.5)

    print("\n" + "="*40)
    print("  Speleo-X Keyboard Teleop")
    print("="*40)
    print("  W/↑  — Forward")
    print("  S/↓  — Reverse")
    print("  A/←  — Turn Left")
    print("  D/→  — Turn Right")
    print("  SPACE — Stop")
    print("  Q    — Quit")
    print("="*40 + "\n")

    try:
        while True:
            key = get_key()

            if key == 'q' or key == '\x03':  # Q or Ctrl+C
                print("\nStopping motors and exiting...")
                send(ser, 0, 0)
                break

            if key in COMMANDS:
                left, right, label = COMMANDS[key]
                send(ser, left, right)
                print(f"  >> {label}  (L={left:+d}  R={right:+d})")
            else:
                # Any other key = stop
                send(ser, 0, 0)
                print(f"  >> STOP (unknown key)")

    except KeyboardInterrupt:
        print("\nStopping motors...")
        send(ser, 0, 0)
    finally:
        ser.close()


if __name__ == '__main__':
    main()
