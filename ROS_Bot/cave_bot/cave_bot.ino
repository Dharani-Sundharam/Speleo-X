#include <AFMotor.h>
#include <Wire.h>

const int MPU_ADDR = 0x68;
AF_DCMotor motorFrontLeft(1);
AF_DCMotor motorRearLeft(2);
AF_DCMotor motorFrontRight(3);
AF_DCMotor motorRearRight(4);

unsigned long lastImuTime = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // --- THE ANTI-FREEZE PATCH ---
  // Forces the I2C bus to reset after 3000 microseconds (3ms) if jammed by motor noise
  Wire.setWireTimeout(3000, true); 
  
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x6B); Wire.write(0); Wire.endTransmission(true);
  
  // Start with brakes on
  motorFrontLeft.run(RELEASE); motorRearLeft.run(RELEASE);
  motorFrontRight.run(RELEASE); motorRearRight.run(RELEASE);
}

void loop() {
  unsigned long now = millis();

  // --- 1. SEND GYRO (Required so ROS doesn't crash) ---
  if (now - lastImuTime >= 20) {
    lastImuTime = now;
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x47); Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 2, true);
    float gz_rad = ((Wire.read() << 8 | Wire.read()) / 131.0) * 0.0174533;
    Serial.print("gz:"); Serial.println(gz_rad, 5);
  }

  // --- 2. NO-BS MOTOR COMMANDS ---
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim(); // Annihilate any invisible hidden characters

    // EXACT MATCH: FULL STOP
    if (data == "m,0,0") {
      motorFrontLeft.setSpeed(0);  motorFrontLeft.run(RELEASE);
      motorRearLeft.setSpeed(0);   motorRearLeft.run(RELEASE);
      motorFrontRight.setSpeed(0); motorFrontRight.run(RELEASE);
      motorRearRight.setSpeed(0);  motorRearRight.run(RELEASE);
    }
    // EXACT MATCH: FORWARD
    else if (data == "m,200,200") {
      motorFrontLeft.setSpeed(200);  motorFrontLeft.run(FORWARD);
      motorRearLeft.setSpeed(200);   motorRearLeft.run(FORWARD);
      motorFrontRight.setSpeed(200); motorFrontRight.run(FORWARD);
      motorRearRight.setSpeed(200);  motorRearRight.run(FORWARD);
    }
    // EXACT MATCH: BACKWARD
    else if (data == "m,-200,-200") {
      motorFrontLeft.setSpeed(200);  motorFrontLeft.run(BACKWARD);
      motorRearLeft.setSpeed(200);   motorRearLeft.run(BACKWARD);
      motorFrontRight.setSpeed(200); motorFrontRight.run(BACKWARD);
      motorRearRight.setSpeed(200);  motorRearRight.run(BACKWARD);
    }
    // CATCH-ALL FOR TURNING (Fallback basic parsing)
    else if (data.startsWith("m,")) {
      int c1 = data.indexOf(',');
      int c2 = data.indexOf(',', c1 + 1);
      if (c1 != -1 && c2 != -1) {
        int L = data.substring(c1 + 1, c2).toInt();
        int R = data.substring(c2 + 1).toInt();
        
        if (L > 0) { motorFrontLeft.setSpeed(L); motorFrontLeft.run(FORWARD); motorRearLeft.setSpeed(L); motorRearLeft.run(FORWARD); }
        else { motorFrontLeft.setSpeed(-L); motorFrontLeft.run(BACKWARD); motorRearLeft.setSpeed(-L); motorRearLeft.run(BACKWARD); }
        
        if (R > 0) { motorFrontRight.setSpeed(R); motorFrontRight.run(FORWARD); motorRearRight.setSpeed(R); motorRearRight.run(FORWARD); }
        else { motorFrontRight.setSpeed(-R); motorFrontRight.run(BACKWARD); motorRearRight.setSpeed(-R); motorRearRight.run(BACKWARD); }
      }
    }
  }
}
