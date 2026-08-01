#!/usr/bin/env python3
"""
gyro_logger.py - Reads raw gz (gyro Z) data from the Arduino and logs it.
Run this WHILE the LiDARs are spinning to measure vibration-induced gyro noise.
Run it AGAIN with LiDARs off to see the baseline drift.

Usage:
  python3 gyro_logger.py
"""

import serial
import time
import statistics

PORT      = '/dev/arduino'
BAUD_RATE = 115200
LOG_SECS  = 15  # collect for this many seconds then print stats

print(f"Opening {PORT} at {BAUD_RATE} baud...")
print(f"Collecting gyro data for {LOG_SECS} seconds...")
print("Keep the bot STATIONARY!\n")

readings = []
start = time.time()

try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
    while time.time() - start < LOG_SECS:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith('gz:'):
            try:
                gz = float(line[3:])
                readings.append(gz)
                elapsed = time.time() - start
                print(f"[{elapsed:5.1f}s]  gz = {gz:+.6f} rad/s")
            except ValueError:
                pass

    ser.close()
except serial.SerialException as e:
    print(f"ERROR: Could not open serial port: {e}")
    exit(1)

if len(readings) < 5:
    print("Not enough readings collected! Check your serial port.")
    exit(1)

print("\n" + "="*50)
print("RESULTS (bot stationary):")
print(f"  Samples        : {len(readings)}")
print(f"  Mean gz        : {statistics.mean(readings):+.6f} rad/s   <-- bias offset")
print(f"  Std deviation  : {statistics.stdev(readings):.6f} rad/s   <-- noise floor")
print(f"  Min gz         : {min(readings):+.6f} rad/s")
print(f"  Max gz         : {max(readings):+.6f} rad/s")
print(f"  Peak-to-peak   : {max(readings) - min(readings):.6f} rad/s")
print("="*50)
print("\nRecommended dead-zone threshold: anything |gz| below the std deviation")
print(f"  Suggested threshold = {statistics.stdev(readings) * 2:.4f} rad/s")
