/**
 * Speleo-X — STM32F103C8 Blue Pill  ROS Serial Bridge
 * =====================================================
 *
 * Upgraded from hw_verify test sketch to a full ROS 2 serial bridge.
 * The Raspberry Pi controls the motors and receives sensor data over
 * USART1 (PA9/PA10) at 115200 baud.
 *
 * ── Serial Protocol ──────────────────────────────────────────────────────────
 *
 *   IN  (Pi → STM32):
 *     "m,<left_pwm>,<right_pwm>\n"
 *       left_pwm, right_pwm : signed integer, range −255 … +255
 *       Positive = forward,  Negative = reverse,  0 = stop
 *     Example: "m,200,200\n"  → both motors forward at ~78% power
 *
 *   OUT (STM32 → Pi):   published at 20 Hz (every 50 ms)
 *     "T=<ms> ENC_L=<ticks> ENC_R=<ticks> AX=<raw> AY=<raw> AZ=<raw> GX=<raw> GY=<raw> GZ=<raw>"
 *       ENC_L / ENC_R : cumulative encoder ticks (signed long)
 *       AX/AY/AZ      : MPU6050 raw accelerometer (±2 g full-scale → LSB = 16384/g)
 *       GX/GY/GZ      : MPU6050 raw gyroscope     (±250 °/s full-scale → LSB = 131/(°/s))
 *
 *   Comment lines start with '#' and are safe to ignore on the Pi side.
 *
 * ── Pin map ──────────────────────────────────────────────────────────────────
 *   Left  motor driver (Board 1) : PB8 = IN1 (FWD PWM), PB9  = IN2 (REV PWM)
 *   Right motor driver (Board 2) : PB0 = IN1 (FWD PWM), PB10 = IN2 (REV PWM)
 *                                  (PB1 was hardware-stuck HIGH — moved to PB10)
 *   Left  encoder                : PA0 = Channel A (EXTI), PA1 = Channel B
 *   Right encoder                : PA2 = Channel A (EXTI), PA3 = Channel B
 *   I2C / MPU6050                : PB6 = SCL, PB7 = SDA  (address 0x68)
 *   Serial telemetry             : USART1 PA9 (TX) / PA10 (RX) @ 115200
 *
 * ── Build notes (Arduino IDE + stm32duino) ───────────────────────────────────
 *   Board            : Generic STM32F1 series
 *   Board part number: BluePill F103C8
 *   U(S)ART support  : Enabled (generic 'Serial')
 *   Upload method    : STM32CubeProgrammer (SWD) or STLink
 *
 * ── PlatformIO env ───────────────────────────────────────────────────────────
 *   [env:bluepill_f103c8]
 *   platform         = ststm32
 *   board            = bluepill_f103c8
 *   framework        = arduino
 *   upload_protocol  = stlink
 */

#include <Wire.h>

// ── Motor control pins (DRV8833 IN1/IN2, both PWM-capable timer pins) ────────
static const uint8_t LEFT_MOTOR_FWD  = PB8;   // TIM4_CH3
static const uint8_t LEFT_MOTOR_REV  = PB9;   // TIM4_CH4
static const uint8_t RIGHT_MOTOR_FWD = PB0;   // TIM3_CH3
static const uint8_t RIGHT_MOTOR_REV = PB10;  // TIM2_CH3  (PB1 was stuck HIGH)

// ── Quadrature encoder pins ───────────────────────────────────────────────────
static const uint8_t LEFT_ENCODER_A  = PA0;   // EXTI0
static const uint8_t LEFT_ENCODER_B  = PA1;
static const uint8_t RIGHT_ENCODER_A = PA2;   // EXTI2 — no I2C conflict
static const uint8_t RIGHT_ENCODER_B = PA3;

// ── MPU6050 I2C register map ──────────────────────────────────────────────────
static const uint8_t I2C_SCL_PIN        = PB6;
static const uint8_t I2C_SDA_PIN        = PB7;
static const uint8_t MPU6050_ADDR       = 0x68;
static const uint8_t MPU6050_REG_PWR    = 0x6B;  // Power management 1
static const uint8_t MPU6050_REG_ACCEL  = 0x3B;  // ACCEL_XOUT_H (6 bytes)
static const uint8_t MPU6050_REG_GYRO   = 0x43;  // GYRO_XOUT_H  (6 bytes)

// ── Timing ────────────────────────────────────────────────────────────────────
static const uint32_t TELEMETRY_INTERVAL_MS = 50;    // 20 Hz output
static const uint32_t WATCHDOG_TIMEOUT_MS   = 1000;  // stop if Pi silent > 1 s

// ── Encoder tick counters — volatile, read under interrupt mask ───────────────
volatile long left_encoder_ticks  = 0;
volatile long right_encoder_ticks = 0;

// ── Motor state (for watchdog comparison) ────────────────────────────────────
static int current_left_pwm  = 0;
static int current_right_pwm = 0;

// ── Timestamps ────────────────────────────────────────────────────────────────
static uint32_t last_telemetry_ms = 0;
static uint32_t last_command_ms   = 0;

// ── Serial input buffer ───────────────────────────────────────────────────────
static String serial_buffer = "";

// =============================================================================
// Quadrature encoder ISRs
// =============================================================================
void leftEncoderISR() {
  if (digitalRead(LEFT_ENCODER_A) == digitalRead(LEFT_ENCODER_B))
    left_encoder_ticks++;
  else
    left_encoder_ticks--;
}

void rightEncoderISR() {
  if (digitalRead(RIGHT_ENCODER_A) == digitalRead(RIGHT_ENCODER_B))
    right_encoder_ticks++;
  else
    right_encoder_ticks--;
}

// =============================================================================
// Motor control — signed PWM  (−255 … +255)
// Right motor is mounted mirrored: positive PWM → REV pin drives it forward.
// =============================================================================
static void setMotors(int left, int right) {
  left  = constrain(left,  -255, 255);
  right = constrain(right, -255, 255);
  current_left_pwm  = left;
  current_right_pwm = right;

  // Left motor
  if (left > 0) {
    analogWrite(LEFT_MOTOR_FWD, left);
    analogWrite(LEFT_MOTOR_REV, 0);
  } else if (left < 0) {
    analogWrite(LEFT_MOTOR_FWD, 0);
    analogWrite(LEFT_MOTOR_REV, -left);
  } else {
    analogWrite(LEFT_MOTOR_FWD, 0);
    analogWrite(LEFT_MOTOR_REV, 0);
  }

  // Right motor — physically mirrored, so FWD/REV signals are swapped.
  if (right > 0) {
    analogWrite(RIGHT_MOTOR_FWD, 0);
    analogWrite(RIGHT_MOTOR_REV, right);
  } else if (right < 0) {
    analogWrite(RIGHT_MOTOR_FWD, -right);
    analogWrite(RIGHT_MOTOR_REV, 0);
  } else {
    analogWrite(RIGHT_MOTOR_FWD, 0);
    analogWrite(RIGHT_MOTOR_REV, 0);
  }
}

// =============================================================================
// MPU6050 helpers
// =============================================================================
static bool mpu6050WriteReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

// Read 3 consecutive 16-bit big-endian signed registers starting at 'reg'.
static bool mpu6050Read6(uint8_t reg, int16_t &a, int16_t &b, int16_t &c) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(static_cast<int>(MPU6050_ADDR), 6) != 6) return false;

  const uint8_t ah = Wire.read(), al = Wire.read();
  const uint8_t bh = Wire.read(), bl = Wire.read();
  const uint8_t ch = Wire.read(), cl = Wire.read();

  a = static_cast<int16_t>((ah << 8) | al);
  b = static_cast<int16_t>((bh << 8) | bl);
  c = static_cast<int16_t>((ch << 8) | cl);
  return true;
}

// =============================================================================
// Serial command parser — expects "m,<left>,<right>\n"
// =============================================================================
static void parseCommand(const String &line) {
  if (!line.startsWith(F("m,"))) return;

  const int c1 = line.indexOf(',');
  const int c2 = line.indexOf(',', c1 + 1);
  if (c1 < 0 || c2 < 0) {
    Serial.println(F("# ERR: bad format, expected m,<left>,<right>"));
    return;
  }

  const int left_pwm  = line.substring(c1 + 1, c2).toInt();
  const int right_pwm = line.substring(c2 + 1).toInt();

  setMotors(left_pwm, right_pwm);
  last_command_ms = millis();
}

// =============================================================================
// Telemetry output — sent at 20 Hz
// Format: "T=<ms> ENC_L=<ticks> ENC_R=<ticks> AX=.. AY=.. AZ=.. GX=.. GY=.. GZ=.."
// =============================================================================
static void printTelemetry() {
  // Read IMU
  int16_t ax = 0, ay = 0, az = 0;
  int16_t gx = 0, gy = 0, gz = 0;
  const bool accel_ok = mpu6050Read6(MPU6050_REG_ACCEL, ax, ay, az);
  const bool gyro_ok  = mpu6050Read6(MPU6050_REG_GYRO,  gx, gy, gz);

  // Atomic snapshot of encoder ticks
  long enc_l, enc_r;
  noInterrupts();
  enc_l = left_encoder_ticks;
  enc_r = right_encoder_ticks;
  interrupts();

  // Odometry
  Serial.print(F("T="));     Serial.print(millis());
  Serial.print(F(" ENC_L=")); Serial.print(enc_l);
  Serial.print(F(" ENC_R=")); Serial.print(enc_r);

  // Accelerometer (±2g, LSB = 16384 counts/g)
  Serial.print(F(" AX="));   Serial.print(accel_ok ? ax : 0);
  Serial.print(F(" AY="));   Serial.print(accel_ok ? ay : 0);
  Serial.print(F(" AZ="));   Serial.print(accel_ok ? az : 0);

  // Gyroscope (±250°/s, LSB = 131 counts per °/s)
  Serial.print(F(" GX="));   Serial.print(gyro_ok  ? gx : 0);
  Serial.print(F(" GY="));   Serial.print(gyro_ok  ? gy : 0);
  Serial.print(F(" GZ="));   Serial.println(gyro_ok ? gz : 0);
}

// =============================================================================
// Arduino entry points
// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { /* wait for USB-UART */ }

  Serial.println(F("# Speleo-X ROS Bridge — STM32F103C8 Blue Pill"));
  Serial.println(F("# Command : m,<left_pwm>,<right_pwm>  (range -255..+255)"));
  Serial.println(F("# Telemetry: T ENC_L ENC_R AX AY AZ GX GY GZ  @ 20 Hz"));

  // ── Motor pins
  pinMode(LEFT_MOTOR_FWD,  OUTPUT);
  pinMode(LEFT_MOTOR_REV,  OUTPUT);
  pinMode(RIGHT_MOTOR_FWD, OUTPUT);
  pinMode(RIGHT_MOTOR_REV, OUTPUT);
  setMotors(0, 0);
  Serial.println(F("# Motors  : ready (all stopped)"));

  // ── Encoder pins + interrupts
  pinMode(LEFT_ENCODER_A,  INPUT_PULLUP);
  pinMode(LEFT_ENCODER_B,  INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_A, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_A),  leftEncoderISR,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_A), rightEncoderISR, CHANGE);
  Serial.println(F("# Encoders: PA0/PA1 (L), PA2/PA3 (R)"));

  // ── I2C / MPU6050
  Wire.setSCL(I2C_SCL_PIN);
  Wire.setSDA(I2C_SDA_PIN);
  Wire.begin();
  Wire.setClock(400000);  // 400 kHz fast mode

  if (mpu6050WriteReg(MPU6050_REG_PWR, 0x00)) {
    Serial.println(F("# MPU6050 : awake — accel + gyro active"));
  } else {
    Serial.println(F("# MPU6050 : I2C FAILED — check PB6/PB7 wiring"));
  }

  last_telemetry_ms = millis();
  last_command_ms   = millis();

  Serial.println(F("# Ready. Waiting for commands..."));
}

void loop() {
  const uint32_t now = millis();

  // ── Read incoming serial (non-blocking, line-buffered)
  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      serial_buffer.trim();
      if (serial_buffer.length() > 0) {
        parseCommand(serial_buffer);
        serial_buffer = "";
      }
    } else {
      serial_buffer += c;
    }
  }

  // ── Watchdog: stop motors if Pi has gone silent
  if ((now - last_command_ms > WATCHDOG_TIMEOUT_MS) &&
      (current_left_pwm != 0 || current_right_pwm != 0)) {
    setMotors(0, 0);
    Serial.println(F("# Watchdog: no command for 1s — motors stopped"));
  }

  // ── Telemetry at 20 Hz
  if (now - last_telemetry_ms >= TELEMETRY_INTERVAL_MS) {
    last_telemetry_ms = now;
    printTelemetry();
  }
}
