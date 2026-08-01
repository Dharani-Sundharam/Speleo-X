#!/usr/bin/env python3
"""
speleo_bridge.py — Speleo-X STM32 Blue Pill ↔ ROS 2 Serial Bridge
===================================================================

Matches the firmware serial protocol in stm32_bluepill_hw_verify.ino:

  IN  (Pi → STM32) : "m,<left_pwm>,<right_pwm>\n"   (-255 to +255)
  OUT (STM32 → Pi) : "T=<ms> ENC_L=<t> ENC_R=<t> AX=<r> AY=<r> AZ=<r> GX=<r> GY=<r> GZ=<r>"

Published ROS 2 Topics:
  /odom      nav_msgs/Odometry      — encoder-based dead-reckoning odometry
  /imu/raw   sensor_msgs/Imu        — raw MPU6050 accel + gyro (SI units)
  /enc_ticks std_msgs/String        — raw encoder tick string (debug)

Subscribed ROS 2 Topics:
  /cmd_vel   geometry_msgs/Twist    — controls motors

Usage:
  python3 speleo_bridge.py

Parameters (edit below):
  SERIAL_PORT   — serial device (default: /dev/bluepill)
  BAUD_RATE     — 115200 (must match firmware)
  WHEEL_RADIUS  — metres, N20 wheel radius
  WHEEL_BASE    — metres, distance between left and right wheels
  ACCEL_SCALE   — MPU6050 ±2g  → 16384 LSB/g
  GYRO_SCALE    — MPU6050 ±250°/s → 131 LSB/(°/s)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

import serial
import threading
import math
import time

# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT  = '/dev/bluepill'   # or /dev/ttyUSB0
BAUD_RATE    = 115200
WHEEL_RADIUS     = 0.035   # metres — wheel radius (35 mm)
WHEEL_BASE       = 0.20    # metres — centre-to-centre wheel separation
TICKS_PER_REV_L  = 725    # measured ticks per full revolution, LEFT  wheel
TICKS_PER_REV_R  = 711    # measured ticks per full revolution, RIGHT wheel

# MPU6050 conversion constants
ACCEL_SCALE  = 16384.0          # LSB per g   (±2g mode)
GYRO_SCALE   = 131.0            # LSB per °/s (±250°/s mode)
GRAVITY      = 9.81             # m/s²

# cmd_vel → PWM mapping
MAX_LINEAR_MPS  = 0.3           # m/s at full PWM 255
MAX_ANGULAR_RPS = 2.0           # rad/s at full PWM 255
WATCHDOG_STOP_SEC = 1.0         # stop motors if no cmd_vel for this long


class SpeleoXBridge(Node):

    def __init__(self):
        super().__init__('speleo_x_bridge')

        # ── Serial port ──────────────────────────────────────────────────────
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.get_logger().info(f'Serial opened: {SERIAL_PORT} @ {BAUD_RATE}')
        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open serial port {SERIAL_PORT}: {e}')
            raise

        self._lock = threading.Lock()

        # ── Publishers ───────────────────────────────────────────────────────
        self.odom_pub  = self.create_publisher(Odometry, '/odom',     10)
        self.imu_pub   = self.create_publisher(Imu,      '/imu/raw',  10)
        self.enc_pub   = self.create_publisher(String,   '/enc_ticks', 10)
        self.tf_br     = TransformBroadcaster(self)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)

        # ── State ────────────────────────────────────────────────────────────
        self._prev_enc_l = None
        self._prev_enc_r = None
        self._x   = 0.0
        self._y   = 0.0
        self._th  = 0.0
        self._vx  = 0.0
        self._wz  = 0.0
        self._prev_t = None

        self._last_cmd_time = time.time()
        self._cur_pwm = (0, 0)

        # ── Watchdog timer ───────────────────────────────────────────────────
        self.create_timer(0.2, self._watchdog)

        # ── Serial reader thread ─────────────────────────────────────────────
        t = threading.Thread(target=self._read_serial_loop, daemon=True)
        t.start()

        self.get_logger().info('Speleo-X bridge ready. Listening on /cmd_vel...')

    # =========================================================================
    # Serial reader — runs in background thread
    # =========================================================================
    def _read_serial_loop(self):
        while rclpy.ok():
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()

                # Skip comment lines (start with #)
                if not line or line.startswith('#'):
                    continue

                self._parse_telemetry(line)

            except Exception as e:
                self.get_logger().warn(f'Serial read error: {e}', throttle_duration_sec=5.0)

    # =========================================================================
    # Parse firmware telemetry line
    # "T=1234 ENC_L=450 ENC_R=448 AX=-102 AY=44 AZ=16320 GX=8 GY=-3 GZ=15"
    # =========================================================================
    def _parse_telemetry(self, line: str):
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

            # Convert raw IMU values to SI units
            ax = fields['AX'] / ACCEL_SCALE * GRAVITY   # m/s²
            ay = fields['AY'] / ACCEL_SCALE * GRAVITY
            az = fields['AZ'] / ACCEL_SCALE * GRAVITY
            gx = math.radians(fields['GX'] / GYRO_SCALE)  # rad/s
            gy = math.radians(fields['GY'] / GYRO_SCALE)
            gz = math.radians(fields['GZ'] / GYRO_SCALE)

            now  = self.get_clock().now()
            stamp = now.to_msg()

            # ── Odometry (differential drive dead-reckoning) ─────────────────
            if self._prev_enc_l is not None and self._prev_t is not None:
                d_enc_l = enc_l - self._prev_enc_l
                d_enc_r = enc_r - self._prev_enc_r

                # Ticks → metres
                d_left  = (d_enc_l / TICKS_PER_REV_L) * 2 * math.pi * WHEEL_RADIUS
                d_right = (d_enc_r / TICKS_PER_REV_R) * 2 * math.pi * WHEEL_RADIUS
                d_center = (d_left + d_right) / 2.0
                d_theta  = (d_right - d_left) / WHEEL_BASE

                self._th += d_theta
                self._th  = math.atan2(math.sin(self._th), math.cos(self._th))
                self._x  += d_center * math.cos(self._th)
                self._y  += d_center * math.sin(self._th)

                # Velocity estimate (from gyro Z for angular)
                dt = (now.nanoseconds - self._prev_t) * 1e-9
                if dt > 0:
                    self._vx = d_center / dt
                    self._wz = gz   # use IMU gyro Z for angular velocity

            self._prev_enc_l = enc_l
            self._prev_enc_r = enc_r
            self._prev_t     = now.nanoseconds

            # ── Quaternion from yaw ──────────────────────────────────────────
            q = self._yaw_to_quat(self._th)

            # ── Publish /odom ────────────────────────────────────────────────
            odom = Odometry()
            odom.header.stamp    = stamp
            odom.header.frame_id = 'odom'
            odom.child_frame_id  = 'base_link'
            odom.pose.pose.position.x  = self._x
            odom.pose.pose.position.y  = self._y
            odom.pose.pose.orientation = q
            odom.twist.twist.linear.x  = self._vx
            odom.twist.twist.angular.z = self._wz
            # Diagonal covariance — tune these once robot is calibrated
            odom.pose.covariance[0]  = 0.01
            odom.pose.covariance[7]  = 0.01
            odom.pose.covariance[35] = 0.05
            odom.twist.covariance[0]  = 0.01
            odom.twist.covariance[35] = 0.05
            self.odom_pub.publish(odom)

            # ── Publish TF odom → base_link ─────────────────────────────────
            tf = TransformStamped()
            tf.header.stamp    = stamp
            tf.header.frame_id = 'odom'
            tf.child_frame_id  = 'base_link'
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.translation.z = 0.0
            tf.transform.rotation      = q
            self.tf_br.sendTransform(tf)

            # ── Publish /imu/raw ─────────────────────────────────────────────
            imu = Imu()
            imu.header.stamp    = stamp
            imu.header.frame_id = 'base_link'
            imu.linear_acceleration.x = ax
            imu.linear_acceleration.y = ay
            imu.linear_acceleration.z = az
            imu.angular_velocity.x    = gx
            imu.angular_velocity.y    = gy
            imu.angular_velocity.z    = gz
            imu.orientation           = q
            # Covariance — -1 means "not calibrated"
            imu.orientation_covariance[0]          = -1.0
            imu.linear_acceleration_covariance[0]  =  0.1
            imu.angular_velocity_covariance[0]     =  0.01
            self.imu_pub.publish(imu)

            # ── Publish raw encoder ticks (for debug in Foxglove) ────────────
            enc_msg = String()
            enc_msg.data = f'ENC_L={enc_l} ENC_R={enc_r}'
            self.enc_pub.publish(enc_msg)

        except Exception as e:
            self.get_logger().warn(f'Parse error on line "{line}": {e}',
                                   throttle_duration_sec=5.0)

    # =========================================================================
    # cmd_vel → motor PWM command
    # =========================================================================
    def _on_cmd_vel(self, msg: Twist):
        self._last_cmd_time = time.time()

        vx = msg.linear.x
        wz = msg.angular.z

        # Scale to PWM range -255..+255
        left_pwm  = int(vx / MAX_LINEAR_MPS  * 255 - wz / MAX_ANGULAR_RPS * 255)
        right_pwm = int(vx / MAX_LINEAR_MPS  * 255 + wz / MAX_ANGULAR_RPS * 255)

        left_pwm  = max(-255, min(255, left_pwm))
        right_pwm = max(-255, min(255, right_pwm))

        self._send_motor_cmd(left_pwm, right_pwm)

    # =========================================================================
    # Watchdog — stops motors if /cmd_vel goes silent
    # =========================================================================
    def _watchdog(self):
        if (time.time() - self._last_cmd_time > WATCHDOG_STOP_SEC and
                self._cur_pwm != (0, 0)):
            self.get_logger().info('Watchdog: stopping motors (no cmd_vel)')
            self._send_motor_cmd(0, 0)

    # =========================================================================
    # Write motor command to serial
    # =========================================================================
    def _send_motor_cmd(self, left: int, right: int):
        self._cur_pwm = (left, right)
        cmd = f'm,{left},{right}\n'.encode()
        with self._lock:
            try:
                self.ser.write(cmd)
            except Exception as e:
                self.get_logger().error(f'Serial write error: {e}')

    # =========================================================================
    # Quaternion from yaw angle
    # =========================================================================
    @staticmethod
    def _yaw_to_quat(yaw: float):
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def destroy_node(self):
        self._send_motor_cmd(0, 0)
        self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SpeleoXBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
