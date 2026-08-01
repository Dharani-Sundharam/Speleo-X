#!/usr/bin/env python3
"""
real_bot.py — Cave Bot Bridge (ROS 2 Humble)

Rule:
  cmd_vel x=1   → m,200,200    (forward)
  cmd_vel x=-1  → m,-200,-200  (backward)
  cmd_vel z=1   → m,-200,200   (rotate left)
  cmd_vel z=-1  → m,200,-200   (rotate right)
  No cmd_vel for >1 second → m,0,0 (stop)

Also publishes /odom and /imu.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
import serial, math, time, threading

# ── Settings ─────────────────────────────────────────────────────────────────
SERIAL_PORT   = '/dev/arduino'
BAUD_RATE     = 115200
DRIVE_PWM     = 200          # PWM sent for forward/back
TURN_PWM      = 200          # PWM per wheel for rotation
STOP_AFTER    = 1.0          # seconds of silence → send stop
WHEEL_DIA     = 0.07         # metres (for odom speed estimate)
MOTOR_RPM     = 100          # at full PWM (upgraded from 10 RPM motors)
WHEEL_BASE    = 0.25         # metres
GZ_SIGN       = 1            # 1 = Left turn is positive Z in ROS
GZ_SCALE      = 3.6          # Increased from 1.8 for 100 RPM motors (faster turns need higher scale)
GZ_BIAS       = -0.03528     # Measured raw bias (-0.0098) × GZ_SCALE (3.6) — applied to scaled value
GZ_DEADZONE   = 0.05         # Ignore |corrected gz| below this — covers bias error + vibration noise
GRAVITY       = 9.81

MAX_SPEED_MPS = 0.366  # π × 0.07m × 100 RPM / 60 = 0.366 m/s at full PWM=255
MAX_SPEED_ROT = 3.0    # approx max rad/s at full PWM with 100 RPM motors


class CaveBotBridge(Node):
    def __init__(self):
        super().__init__('cave_bot_bridge')

        # Serial port
        self.ser  = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        self._lock = threading.Lock()
        self.get_logger().info(f'Serial open on {SERIAL_PORT}')

        # ROS publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.imu_pub  = self.create_publisher(Imu,      '/imu',  10)
        self.tf_br    = TransformBroadcaster(self)

        # cmd_vel subscriber
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)

        # State
        self._last_cmd = 0.0        # timestamp of last cmd_vel
        self._cur_pwm  = (0, 0)     # last sent (left, right) PWM
        self._gz  = 0.0             # gyro Z rad/s
        self._ax  = 0.0             # accel X m/s²
        self._ay  = 0.0             # accel Y m/s²
        self._x   = 0.0             # odom position
        self._y   = 0.0
        self._th  = 0.0             # yaw
        self._t   = time.time()

        # Watchdog: sends stop if no cmd_vel for STOP_AFTER seconds
        self.create_timer(0.2, self._watchdog)   # check 5x/sec
        # Odom + IMU publish at 50 Hz (increased from 20Hz for 100RPM fast turns)
        self.create_timer(0.02, self._odom_tick)

        # Serial reader in background thread
        threading.Thread(target=self._read_serial, daemon=True).start()

    # ── Direct cmd_vel → serial ───────────────────────────────────────────────
    def _on_cmd_vel(self, msg: Twist):
        self._last_cmd = time.time()

        lx = msg.linear.x
        az = msg.angular.z if msg.angular.z != 0.0 else -msg.linear.y

        if   lx > 0.001:  lpwm, rpwm =  DRIVE_PWM,  DRIVE_PWM   # forward
        elif lx < -0.001: lpwm, rpwm = -DRIVE_PWM, -DRIVE_PWM   # backward
        elif az > 0.01:   lpwm, rpwm = -TURN_PWM,   TURN_PWM    # rotate left
        elif az < -0.01:  lpwm, rpwm =  TURN_PWM,  -TURN_PWM    # rotate right
        else:             lpwm, rpwm =  0, 0                      # explicit stop

        self._send(lpwm, rpwm)

    # ── Watchdog: stop if silent ──────────────────────────────────────────────
    def _watchdog(self):
        if time.time() - self._last_cmd > STOP_AFTER:
            if self._cur_pwm != (0, 0):
                self.get_logger().info('Watchdog: stopping motors')
                self._send(0, 0)

    # ── Thread-safe serial write ──────────────────────────────────────────────
    def _send(self, left: int, right: int):
        self._cur_pwm = (left, right)
        with self._lock:
            try:
                self.ser.write(f'm,{left},{right}\n'.encode())
            except Exception as e:
                self.get_logger().error(f'Serial: {e}')

    # ── Serial reader: parses gz / ax,ay ─────────────────────────────────────
    def _read_serial(self):
        while rclpy.ok():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('gz:'):
                    self._gz = float(line[3:]) * GZ_SIGN * GZ_SCALE
                elif line.startswith('ax:'):
                    p = line.split(',')
                    self._ax = float(p[0][3:]) * GRAVITY
                    self._ay = float(p[1][3:]) * GRAVITY if len(p) > 1 else 0.0
            except Exception:
                pass

    # ── Odom + TF + IMU (20 Hz) ───────────────────────────────────────────────
    def _odom_tick(self):
        now   = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        dt    = now_s - self._t
        self._t = now_s

        # Yaw from IMU — with bias correction and vibration dead-zone
        gz_corrected = self._gz - GZ_BIAS          # subtract constant sensor offset
        if abs(gz_corrected) < GZ_DEADZONE:        # ignore vibration-level noise
            gz_corrected = 0.0
        self._th += gz_corrected * dt
        self._th  = math.atan2(math.sin(self._th), math.cos(self._th))

        # Linear speed from PWM
        avg_pwm = (self._cur_pwm[0] + self._cur_pwm[1]) / 2.0
        vx = math.copysign(abs(avg_pwm) / 255.0 * MAX_SPEED_MPS, avg_pwm)
        self._x += vx * math.cos(self._th) * dt
        self._y += vx * math.sin(self._th) * dt

        stamp = now.to_msg()
        q     = self._yaw_q(self._th)

        # TF
        t = TransformStamped()
        t.header.stamp = stamp; t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.rotation = q
        self.tf_br.sendTransform(t)

        # /odom
        o = Odometry()
        o.header.stamp = stamp; o.header.frame_id = 'odom'
        o.child_frame_id = 'base_link'
        o.pose.pose.position.x = self._x
        o.pose.pose.position.y = self._y
        o.pose.pose.orientation = q
        o.twist.twist.linear.x  = vx
        o.twist.twist.angular.z = self._gz
        self.odom_pub.publish(o)

        # /imu
        pitch = math.atan2(self._ax, GRAVITY)
        roll  = math.atan2(self._ay, GRAVITY)
        imu   = Imu()
        imu.header.stamp = stamp; imu.header.frame_id = 'base_link'
        imu.orientation = self._euler_q(roll, pitch, self._th)
        imu.orientation_covariance    = [0.01,0.0,0.0,0.0,0.01,0.0,0.0,0.0,0.05]
        imu.angular_velocity.z        = self._gz
        imu.angular_velocity_covariance  = [0.01,0.0,0.0,0.0,0.01,0.0,0.0,0.0,0.01]
        imu.linear_acceleration.x     = self._ax
        imu.linear_acceleration.y     = self._ay
        imu.linear_acceleration_covariance = [0.1,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.1]
        self.imu_pub.publish(imu)

    # ── Quaternion helpers ────────────────────────────────────────────────────
    @staticmethod
    def _yaw_q(yaw):
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.z = math.sin(yaw/2); q.w = math.cos(yaw/2)
        return q

    @staticmethod
    def _euler_q(roll, pitch, yaw):
        from geometry_msgs.msg import Quaternion
        cr,sr = math.cos(roll/2), math.sin(roll/2)
        cp,sp = math.cos(pitch/2),math.sin(pitch/2)
        cy,sy = math.cos(yaw/2), math.sin(yaw/2)
        q = Quaternion()
        q.w = cr*cp*cy+sr*sp*sy; q.x = sr*cp*cy-cr*sp*sy
        q.y = cr*sp*cy+sr*cp*sy; q.z = cr*cp*sy-sr*sp*cy
        return q


def main():
    rclpy.init()
    node = CaveBotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send(0, 0)
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
