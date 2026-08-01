import serial, time
try:
    s = serial.Serial('/dev/arduino', 115200, timeout=1)
    print("Opened /dev/arduino")
    time.sleep(2)  # wait for boot
    print("Testing Motors...")
    s.write(b"m,200,200\n")
    time.sleep(1)
    s.write(b"m,0,0\n")
    s.close()
    print("Test Complete.")
except Exception as e:
    print(f"Error: {e}")
