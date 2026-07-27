# Speleo-X STM32 Blue Pill — ROS Serial Bridge Firmware

> **File:** `firmware/stm32_bluepill_hw_verify/stm32_bluepill_hw_verify.ino`
> **Target MCU:** STM32F103C8T6 (Blue Pill)
> **Framework:** Arduino (stm32duino)
> **Baud Rate:** 115200 — USART1 on PA9 (TX) / PA10 (RX)

---

## Overview

This firmware turns the STM32 Blue Pill into a **ROS 2 Serial Bridge** for the Speleo-X rover. The Raspberry Pi sends motor commands over UART and receives continuous sensor telemetry (encoder odometry + 6DOF IMU) in return.

```
┌─────────────────────┐   UART 115200 baud   ┌────────────────────────┐
│   Raspberry Pi      │ ─────────────────────►│   STM32 Blue Pill      │
│   ROS 2 Humble      │   m,PWM_L,PWM_R\n     │   - DRV8833 x2         │
│   real_bot.py       │◄─────────────────────  │   - N20 Motors x2      │
│                     │  T ENC_L ENC_R ...     │   - Encoders x2        │
│                     │  AX AY AZ GX GY GZ     │   - MPU6050 IMU        │
└─────────────────────┘                        └────────────────────────┘
```

---

## Hardware Wiring

### FTDI / USB-UART Adapter → Blue Pill (UART)

UART is always **crossed** — TX of one side connects to RX of the other.

| FTDI Board Pin | Blue Pill Pin | Direction | Purpose |
| :--- | :--- | :--- | :--- |
| **`TX`** | **`PA10`** | FTDI → STM32 | Send motor commands to STM32 |
| **`RX`** | **`PA9`** | STM32 → FTDI | Receive telemetry from STM32 |
| **`GND`** | **`GND`** | Common | Shared ground (required!) |
| `3.3V` / `5V` | *(optional)* | Power | Can power Blue Pill if needed |

> **Warning:** Most FTDI boards output 5V logic by default. Set your FTDI board to **3.3V mode** if it has a voltage jumper.

---

### Motor Drivers (DRV8833 — 2 separate boards)

| Signal | Blue Pill Pin | Driver Board | DRV8833 Pin |
| :--- | :--- | :--- | :--- |
| Left Motor Forward PWM | **`PB8`** | Board 1 | `IN1` / `AIN1` |
| Left Motor Reverse PWM | **`PB9`** | Board 1 | `IN2` / `AIN2` |
| Right Motor Forward PWM | **`PB0`** | Board 2 | `IN1` / `AIN1` |
| Right Motor Reverse PWM | **`PB10`** | Board 2 | `IN2` / `AIN2` |
| Enable (both boards) | `3.3V` | Both | `STBY` / `ULT` |
| Motor Power | Battery (+) | Both | `VM` |
| Ground | `GND` | Both | `GND` |

> `PB1` (original REV pin for right motor) was found to be **hardware-stuck HIGH** via GPIO diagnostic. Permanently moved to `PB10`.

---

### N20 Encoded Motors

| N20 Pin | Connects To |
| :--- | :--- |
| `M1` | DRV8833 `OUT1` |
| `M2` | DRV8833 `OUT2` |
| `VCC` | STM32 `3.3V` |
| `GND` | Common `GND` |
| **Left `C1`** | **`PA0`** (Left Encoder A — interrupt) |
| **Left `C2`** | **`PA1`** (Left Encoder B) |
| **Right `C1`** | **`PA2`** (Right Encoder A — interrupt) |
| **Right `C2`** | **`PA3`** (Right Encoder B) |

> `PA6`/`PA7` (original right encoder pins) conflicted with `PB6`/`PB7` (I2C) on the STM32 EXTI hardware multiplexer. Moved to `PA2`/`PA3` (EXTI2 — no conflicts).

---

### MPU6050 IMU (I2C)

| MPU6050 Pin | Blue Pill Pin |
| :--- | :--- |
| `SCL` | **`PB6`** |
| `SDA` | **`PB7`** |
| `VCC` | `3.3V` |
| `GND` | `GND` |
| `AD0` | `GND` (sets I2C address to `0x68`) |

---

## Serial Protocol

### Command (Raspberry Pi → STM32)

```
m,<left_pwm>,<right_pwm>\n
```

| Parameter | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `left_pwm` | `int` | `-255 ... +255` | Left motor power. `+` = forward, `-` = reverse |
| `right_pwm` | `int` | `-255 ... +255` | Right motor power. `+` = forward, `-` = reverse |

**Examples:**

| Command | Effect |
| :--- | :--- |
| `m,200,200\n` | Both motors forward at ~78% power |
| `m,-200,-200\n` | Both motors reverse at ~78% power |
| `m,-200,200\n` | Tank turn left |
| `m,200,-200\n` | Tank turn right |
| `m,0,0\n` | Stop all motors |

---

### Telemetry (STM32 → Raspberry Pi) — 20 Hz

```
T=<ms> ENC_L=<ticks> ENC_R=<ticks> AX=<raw> AY=<raw> AZ=<raw> GX=<raw> GY=<raw> GZ=<raw>
```

| Field | Unit | Scale / Notes |
| :--- | :--- | :--- |
| `T` | ms | `millis()` — time since boot |
| `ENC_L` | ticks | Cumulative signed encoder count, left wheel |
| `ENC_R` | ticks | Cumulative signed encoder count, right wheel |
| `AX/AY/AZ` | LSB | Raw accelerometer. Divide by `16384` to get g (±2g range) |
| `GX/GY/GZ` | LSB | Raw gyroscope. Divide by `131` to get deg/s (±250 deg/s range) |

Lines beginning with `#` are informational comments — safely ignore them on the Pi side.

**Example line:**
```
T=5050 ENC_L=1203 ENC_R=1198 AX=-102 AY=44 AZ=16320 GX=8 GY=-3 GZ=15
```

---

## Safety — Watchdog Timer

If the STM32 receives **no motor command for more than 1 second**, it automatically stops both motors and prints:
```
# Watchdog: no command for 1s — motors stopped
```
This prevents the robot from running away if the Pi crashes or the USB cable disconnects.

---

## Build & Flash Instructions

### Arduino IDE
1. Install board package: `Tools -> Board -> Boards Manager -> "STM32 MCU based boards"` by STMicroelectronics
2. Select:
   - **Board:** `Generic STM32F1 series`
   - **Board part number:** `BluePill F103C8`
   - **U(S)ART support:** `Enabled (generic 'Serial')`
   - **Upload method:** `STM32CubeProgrammer (SWD)`
3. Connect ST-Link V2 (`3.3V`, `GND`, `SWDIO`, `SWCLK`) to Blue Pill
4. Click **Upload**

### PlatformIO (`platformio.ini`)
```ini
[env:bluepill_f103c8]
platform         = ststm32
board            = bluepill_f103c8
framework        = arduino
upload_protocol  = stlink
```

---

## Quick Test (without ROS)

### From any Serial Monitor (PuTTY / Arduino IDE / minicom)
```
# View live telemetry:
Open port at 115200 baud — data streams automatically at 20 Hz

# Send a motor command:
m,200,200     <- forward
m,0,0         <- stop
```

### From Raspberry Pi terminal
```bash
# View telemetry stream
cat /dev/ttyUSB0   # or /dev/arduino if udev rule is set up

# Send commands
echo "m,200,200" > /dev/ttyUSB0   # forward
echo "m,0,0"     > /dev/ttyUSB0   # stop
```

---

## Known Hardware Issues (Resolved)

| Issue | Cause | Fix Applied |
| :--- | :--- | :--- |
| Right encoder stuck at 0/1 | `PA6`/`PA7` share EXTI6/7 with I2C pins `PB6`/`PB7` — STM32 AFIO multiplexer conflict | Moved right encoder to `PA2`/`PA3` |
| Right motor only spins one direction | `PB1` physically stuck HIGH (confirmed via GPIO readback diagnostic in firmware) | Moved `RIGHT_MOTOR_REV` to `PB10` |
