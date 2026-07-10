/**
 * Speleo-X — STM32F103C8 Blue Pill Chassis Integration Test
 * ============================================================
 *
 * Exercises dual DRV8833 motor drivers, MPU6050 IMU, and dual quadrature
 * wheel encoders in a repeatable, non-blocking motion sequence while
 * streaming 100 ms telemetry over Serial.
 *
 * Pin map (exact wiring):
 *   Left  motor driver (Board 1) : PB8 = Forward, PB9 = Reverse
 *   Right motor driver (Board 2) : PB0 = Forward, PB1 = Reverse
 *   Left  encoder                : PA0 = Channel A, PA1 = Channel B
 *   Right encoder                : PA6 = Channel A, PA7 = Channel B
 *   I2C / MPU6050                : PB6 = SCL, PB7 = SDA  (address 0x68)
 *   Serial telemetry             : USART1 PA9 (TX) / PA10 (RX) @ 115200
 *
 * Motion test cycle (repeats forever):
 *   1. Forward  2.0 s
 *   2. Stop     1.0 s  (all motor pins LOW)
 *   3. Reverse  2.0 s
 *   4. Stop     1.0 s
 *   5. Turn L   1.5 s  (left reverse, right forward)
 *   6. Stop     1.0 s
 *   7. Turn R   1.5 s  (left forward, right reverse)
 *   8. Stop     3.0 s
 *
 * Build notes (Arduino IDE / PlatformIO + stm32duino):
 *   - Board   : Generic STM32F103C8
 *   - USB     : None (use external USB-UART on PA9/PA10)
 *   - U(S)ART : Enabled (maps Serial to USART1)
 *
 * PlatformIO example env (platformio.ini):
 *   [env:bluepill_f103c8]
 *   platform  = ststm32
 *   board     = bluepill_f103c8
 *   framework = arduino
 *   upload_protocol = stlink
 */

#include <Wire.h>

// ---------------------------------------------------------------------------
// Motor control pins — DRV8833 IN1/IN2 per channel
// ---------------------------------------------------------------------------
static const uint8_t LEFT_MOTOR_FWD  = PB8;
static const uint8_t LEFT_MOTOR_REV  = PB9;
static const uint8_t RIGHT_MOTOR_FWD = PB0;
static const uint8_t RIGHT_MOTOR_REV = PB1;

// ---------------------------------------------------------------------------
// Quadrature encoder pins
// ---------------------------------------------------------------------------
static const uint8_t LEFT_ENCODER_A  = PA0;
static const uint8_t LEFT_ENCODER_B  = PA1;
static const uint8_t RIGHT_ENCODER_A = PA6;
static const uint8_t RIGHT_ENCODER_B = PA7;

// ---------------------------------------------------------------------------
// I2C / MPU6050 register map
// ---------------------------------------------------------------------------
static const uint8_t I2C_SCL_PIN       = PB6;
static const uint8_t I2C_SDA_PIN       = PB7;
static const uint8_t MPU6050_I2C_ADDR  = 0x68;
static const uint8_t MPU6050_REG_PWR   = 0x6B;  // Power management 1
static const uint8_t MPU6050_REG_ACCEL = 0x3B;  // ACCEL_XOUT_H (6 bytes)

// ---------------------------------------------------------------------------
// Timing constants (milliseconds)
// ---------------------------------------------------------------------------
static const uint32_t TELEMETRY_INTERVAL_MS = 100;
static const uint32_t DURATION_FORWARD_MS     = 2000;
static const uint32_t DURATION_REVERSE_MS     = 2000;
static const uint32_t DURATION_TURN_MS        = 1500;
static const uint32_t DURATION_STOP_MS        = 1000;
static const uint32_t DURATION_STOP_LONG_MS   = 3000;

// ---------------------------------------------------------------------------
// Encoder tick counters — updated only inside ISRs; read under interrupt mask
// ---------------------------------------------------------------------------
volatile long left_encoder_ticks  = 0;
volatile long right_encoder_ticks = 0;

// ---------------------------------------------------------------------------
// Motion state machine
// ---------------------------------------------------------------------------
enum MotionPhase : uint8_t {
  PHASE_FORWARD,
  PHASE_STOP_AFTER_FORWARD,
  PHASE_REVERSE,
  PHASE_STOP_AFTER_REVERSE,
  PHASE_TURN_LEFT,
  PHASE_STOP_AFTER_TURN_LEFT,
  PHASE_TURN_RIGHT,
  PHASE_STOP_AFTER_TURN_RIGHT,
};

static MotionPhase current_phase      = PHASE_FORWARD;
static uint32_t    phase_start_ms     = 0;
static uint32_t    last_telemetry_ms  = 0;
static uint32_t    cycle_count        = 0;

// ---------------------------------------------------------------------------
// Lightweight quadrature ISRs — Pin A CHANGE, direction from Pin B
// ---------------------------------------------------------------------------
void leftEncoderISR() {
  if (digitalRead(LEFT_ENCODER_A) == digitalRead(LEFT_ENCODER_B)) {
    left_encoder_ticks++;
  } else {
    left_encoder_ticks--;
  }
}

void rightEncoderISR() {
  if (digitalRead(RIGHT_ENCODER_A) == digitalRead(RIGHT_ENCODER_B)) {
    right_encoder_ticks++;
  } else {
    right_encoder_ticks--;
  }
}

// ---------------------------------------------------------------------------
// Motor helpers
// ---------------------------------------------------------------------------
static void stopAllMotors() {
  digitalWrite(LEFT_MOTOR_FWD,  LOW);
  digitalWrite(LEFT_MOTOR_REV,  LOW);
  digitalWrite(RIGHT_MOTOR_FWD, LOW);
  digitalWrite(RIGHT_MOTOR_REV, LOW);
}

static void driveForward() {
  digitalWrite(LEFT_MOTOR_FWD,  HIGH);
  digitalWrite(LEFT_MOTOR_REV,  LOW);
  digitalWrite(RIGHT_MOTOR_FWD, HIGH);
  digitalWrite(RIGHT_MOTOR_REV, LOW);
}

static void driveReverse() {
  digitalWrite(LEFT_MOTOR_FWD,  LOW);
  digitalWrite(LEFT_MOTOR_REV,  HIGH);
  digitalWrite(RIGHT_MOTOR_FWD, LOW);
  digitalWrite(RIGHT_MOTOR_REV, HIGH);
}

static void turnLeftOnDime() {
  digitalWrite(LEFT_MOTOR_FWD,  LOW);
  digitalWrite(LEFT_MOTOR_REV,  HIGH);
  digitalWrite(RIGHT_MOTOR_FWD, HIGH);
  digitalWrite(RIGHT_MOTOR_REV, LOW);
}

static void turnRightOnDime() {
  digitalWrite(LEFT_MOTOR_FWD,  HIGH);
  digitalWrite(LEFT_MOTOR_REV,  LOW);
  digitalWrite(RIGHT_MOTOR_FWD, LOW);
  digitalWrite(RIGHT_MOTOR_REV, HIGH);
}

// ---------------------------------------------------------------------------
// MPU6050 — native Wire transactions (no third-party IMU library)
// ---------------------------------------------------------------------------
static bool mpu6050WriteRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

static bool mpu6050ReadAccelerometerRaw(int16_t& ax, int16_t& ay, int16_t& az) {
  Wire.beginTransmission(MPU6050_I2C_ADDR);
  Wire.write(MPU6050_REG_ACCEL);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(static_cast<int>(MPU6050_I2C_ADDR), 6) != 6) {
    return false;
  }

  const uint8_t xh = Wire.read();
  const uint8_t xl = Wire.read();
  const uint8_t yh = Wire.read();
  const uint8_t yl = Wire.read();
  const uint8_t zh = Wire.read();
  const uint8_t zl = Wire.read();

  ax = static_cast<int16_t>((xh << 8) | xl);
  ay = static_cast<int16_t>((yh << 8) | yl);
  az = static_cast<int16_t>((zh << 8) | zl);
  return true;
}

// ---------------------------------------------------------------------------
// Telemetry — encoder snapshot uses interrupt masking for atomic 32-bit reads
// ---------------------------------------------------------------------------
static const __FlashStringHelper* phaseLabel(MotionPhase phase) {
  switch (phase) {
    case PHASE_FORWARD:               return F("FORWARD");
    case PHASE_STOP_AFTER_FORWARD:    return F("STOP");
    case PHASE_REVERSE:               return F("REVERSE");
    case PHASE_STOP_AFTER_REVERSE:    return F("STOP");
    case PHASE_TURN_LEFT:             return F("TURN_LEFT");
    case PHASE_STOP_AFTER_TURN_LEFT:  return F("STOP");
    case PHASE_TURN_RIGHT:            return F("TURN_RIGHT");
    case PHASE_STOP_AFTER_TURN_RIGHT: return F("STOP");
    default:                          return F("UNKNOWN");
  }
}

static void printTelemetry() {
  int16_t accel_x = 0;
  int16_t accel_y = 0;
  int16_t accel_z = 0;
  const bool imu_ok = mpu6050ReadAccelerometerRaw(accel_x, accel_y, accel_z);

  long encoder_left = 0;
  long encoder_right = 0;
  noInterrupts();
  encoder_left  = left_encoder_ticks;
  encoder_right = right_encoder_ticks;
  interrupts();

  Serial.print(F("cycle="));
  Serial.print(cycle_count);
  Serial.print(F(" phase="));
  Serial.print(phaseLabel(current_phase));
  Serial.print(F(" t="));
  Serial.print(millis());

  if (imu_ok) {
    Serial.print(F(" ACCEL_X="));
    Serial.print(accel_x);
    Serial.print(F(" ACCEL_Y="));
    Serial.print(accel_y);
    Serial.print(F(" ACCEL_Z="));
    Serial.print(accel_z);
  } else {
    Serial.print(F(" ACCEL=ERR"));
  }

  Serial.print(F(" ENC_L="));
  Serial.print(encoder_left);
  Serial.print(F(" ENC_R="));
  Serial.println(encoder_right);
}

// ---------------------------------------------------------------------------
// Non-blocking motion sequencer
// ---------------------------------------------------------------------------
static void beginPhase(MotionPhase next_phase) {
  current_phase  = next_phase;
  phase_start_ms = millis();

  switch (next_phase) {
    case PHASE_FORWARD:
      driveForward();
      Serial.println(F(">> MOTION: FORWARD (2.0 s)"));
      break;
    case PHASE_STOP_AFTER_FORWARD:
    case PHASE_STOP_AFTER_REVERSE:
    case PHASE_STOP_AFTER_TURN_LEFT:
      stopAllMotors();
      Serial.println(F(">> MOTION: STOP (1.0 s)"));
      break;
    case PHASE_REVERSE:
      driveReverse();
      Serial.println(F(">> MOTION: REVERSE (2.0 s)"));
      break;
    case PHASE_TURN_LEFT:
      turnLeftOnDime();
      Serial.println(F(">> MOTION: TURN LEFT (1.5 s)"));
      break;
    case PHASE_TURN_RIGHT:
      turnRightOnDime();
      Serial.println(F(">> MOTION: TURN RIGHT (1.5 s)"));
      break;
    case PHASE_STOP_AFTER_TURN_RIGHT:
      stopAllMotors();
      Serial.println(F(">> MOTION: STOP (3.0 s, end of cycle)"));
      break;
    default:
      stopAllMotors();
      break;
  }
}

static void advanceMotionStateMachine() {
  const uint32_t elapsed_ms = millis() - phase_start_ms;

  switch (current_phase) {
    case PHASE_FORWARD:
      if (elapsed_ms >= DURATION_FORWARD_MS) {
        beginPhase(PHASE_STOP_AFTER_FORWARD);
      }
      break;

    case PHASE_STOP_AFTER_FORWARD:
      if (elapsed_ms >= DURATION_STOP_MS) {
        beginPhase(PHASE_REVERSE);
      }
      break;

    case PHASE_REVERSE:
      if (elapsed_ms >= DURATION_REVERSE_MS) {
        beginPhase(PHASE_STOP_AFTER_REVERSE);
      }
      break;

    case PHASE_STOP_AFTER_REVERSE:
      if (elapsed_ms >= DURATION_STOP_MS) {
        beginPhase(PHASE_TURN_LEFT);
      }
      break;

    case PHASE_TURN_LEFT:
      if (elapsed_ms >= DURATION_TURN_MS) {
        beginPhase(PHASE_STOP_AFTER_TURN_LEFT);
      }
      break;

    case PHASE_STOP_AFTER_TURN_LEFT:
      if (elapsed_ms >= DURATION_STOP_MS) {
        beginPhase(PHASE_TURN_RIGHT);
      }
      break;

    case PHASE_TURN_RIGHT:
      if (elapsed_ms >= DURATION_TURN_MS) {
        beginPhase(PHASE_STOP_AFTER_TURN_RIGHT);
      }
      break;

    case PHASE_STOP_AFTER_TURN_RIGHT:
      if (elapsed_ms >= DURATION_STOP_LONG_MS) {
        cycle_count++;
        beginPhase(PHASE_FORWARD);
      }
      break;
  }
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // Wait briefly for USB-UART adapter; safe no-op if not connected.
  }

  Serial.println();
  Serial.println(F("Speleo-X STM32 Blue Pill — Chassis Integration Test"));
  Serial.println(F("==================================================="));

  // Motor outputs — chassis must start at full standstill (all LOW).
  pinMode(LEFT_MOTOR_FWD,  OUTPUT);
  pinMode(LEFT_MOTOR_REV,  OUTPUT);
  pinMode(RIGHT_MOTOR_FWD, OUTPUT);
  pinMode(RIGHT_MOTOR_REV, OUTPUT);
  stopAllMotors();
  Serial.println(F("Motors  : OUTPUT, all pins LOW (standstill)"));

  // Encoder inputs with internal pull-ups; EXTI on channel A (CHANGE).
  pinMode(LEFT_ENCODER_A,  INPUT_PULLUP);
  pinMode(LEFT_ENCODER_B,  INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_A, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_A),  leftEncoderISR,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_A), rightEncoderISR, CHANGE);
  Serial.println(F("Encoders: PA0/PA1 (L), PA6/PA7 (R), ISR quadrature on A-edge"));

  // I2C bus on PB6/PB7, wake MPU6050 from sleep.
  Wire.setSCL(I2C_SCL_PIN);
  Wire.setSDA(I2C_SDA_PIN);
  Wire.begin();
  Wire.setClock(100000);

  if (mpu6050WriteRegister(MPU6050_REG_PWR, 0x00)) {
    Serial.println(F("MPU6050 : awake (PWR_MGMT_1 = 0x00)"));
  } else {
    Serial.println(F("MPU6050 : I2C wake FAILED — check wiring"));
  }

  left_encoder_ticks  = 0;
  right_encoder_ticks = 0;
  last_telemetry_ms   = millis();
  phase_start_ms      = millis();
  driveForward();
  Serial.println(F("Sequence: starting FORWARD phase"));
  Serial.println(F("Telemetry interval: 100 ms"));
  Serial.println();
}

void loop() {
  const uint32_t now_ms = millis();

  if (now_ms - last_telemetry_ms >= TELEMETRY_INTERVAL_MS) {
    last_telemetry_ms = now_ms;
    printTelemetry();
  }

  advanceMotionStateMachine();
}
