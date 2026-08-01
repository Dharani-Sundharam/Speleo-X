#!/usr/bin/env python3
"""
laser_stitcher.py - Push-broom 3D mapping node.

Subscribes to /scan_tilted, transforms each sweep into the odom frame,
and accumulates them into a persistent voxel-grid-based PointCloud2.

Voxel grid: only one point is stored per VOXEL_SIZE^3 cube.
This prevents duplicate points when stationary and keeps the map compact.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry import LaserProjection
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.qos import qos_profile_sensor_data
import numpy as np

# One point per VOXEL_SIZE³ metre cube (2cm grid = fine cave detail)
VOXEL_SIZE   = 0.02   # metres  — smaller = higher resolution
MAX_VOXELS   = 500000 # cap on stored voxels (~6 MB per publish, under Foxglove's 20MB limit)


class LaserStitcher(Node):
    def __init__(self):
        super().__init__('laser_stitcher')
        self.declare_parameter('scan_topic',   '/scan_tilted')
        self.declare_parameter('cloud_topic',  '/cloud_stitched')
        self.declare_parameter('global_frame', 'map')  # map frame: SLAM-corrected, no drift!

        scan_topic        = self.get_parameter('scan_topic').value
        cloud_topic       = self.get_parameter('cloud_topic').value
        self.global_frame = self.get_parameter('global_frame').value

        self.projector  = LaserProjection()
        self.tf_buffer  = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointCloud2, cloud_topic, 1)

        # Voxel dictionary: key = (ix, iy, iz) integer grid coords → value = (x,y,z) float
        self.voxels: dict = {}

        self.get_logger().info(
            f"Laser Stitcher started. Stitching {scan_topic} → {cloud_topic} "
            f"in frame '{self.global_frame}' | voxel={VOXEL_SIZE}m | max={MAX_VOXELS}")

    # ── scan callback ─────────────────────────────────────────────────────────
    def scan_cb(self, scan_msg):
        # 1. Project LaserScan → PointCloud2 in sensor frame
        cloud_in = self.projector.projectLaser(scan_msg)

        # 2. Look up TF: sensor frame → global frame
        try:
            trans = self.tf_buffer.lookup_transform(
                self.global_frame,
                scan_msg.header.frame_id,
                rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Laser Stitcher waiting for TF: {e}")
            return

        # 3. Apply transform to all points
        from transforms3d.quaternions import quat2mat
        q = [trans.transform.rotation.w,
             trans.transform.rotation.x,
             trans.transform.rotation.y,
             trans.transform.rotation.z]
        t = [trans.transform.translation.x,
             trans.transform.translation.y,
             trans.transform.translation.z]
        rot_mat = quat2mat(q)

        pts = list(pc2.read_points(cloud_in, skip_nans=True, field_names=("x", "y", "z")))
        if not pts:
            return

        new_points = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        transformed = new_points.dot(rot_mat.T) + t  # shape Nx3

        # 4. Insert into voxel grid (one point per cell, newest wins)
        inv = 1.0 / VOXEL_SIZE
        for pt in transformed:
            key = (int(pt[0] * inv), int(pt[1] * inv), int(pt[2] * inv))
            self.voxels[key] = (float(pt[0]), float(pt[1]), float(pt[2]))

        # 5. If over the cap, drop the oldest entries (dict preserves insertion order in Python 3.7+)
        if len(self.voxels) > MAX_VOXELS:
            overflow = len(self.voxels) - MAX_VOXELS
            for key in list(self.voxels.keys())[:overflow]:
                del self.voxels[key]

        # 6. Publish the full accumulated voxel map
        header = scan_msg.header
        header.frame_id = self.global_frame
        cloud_out = pc2.create_cloud_xyz32(header, list(self.voxels.values()))
        self.pub.publish(cloud_out)


def main(args=None):
    rclpy.init(args=args)
    node = LaserStitcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
