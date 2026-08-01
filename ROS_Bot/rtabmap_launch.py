#!/usr/bin/env python3
"""
rtabmap_launch.py - Launch RTAB-Map for 3D cave mapping.

Inputs:
  /scan           (flat LiDAR - 2D occupancy grid + loop closure)
  /scan_cloud     (assembled tilted LiDAR cloud - 3D mapping)
  /odom           (wheel + gyro odometry)

Outputs:
  /map            (2D occupancy grid - same as slam_toolbox, works with Nav2)
  /rtabmap/cloud_map  (3D point cloud of environment)
  /tf  map → odom correction

Map is saved to ~/cave_map.db. Load with:
  ros2 launch ... localization_mode:=true
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    mode         = LaunchConfiguration('localization', default='false')

    rtabmap_parameters = [{
        'frame_id':           'base_link',
        'odom_frame_id':      'odom',
        'subscribe_scan':     True,         # flat LiDAR for 2D + loop closure
        'subscribe_scan_cloud': True,       # tilted cloud for 3D map

        # ICP-based loop closure (LiDAR only, no camera)
        'Reg/Strategy':       '1',          # 1 = ICP
        'Icp/Iterations':     '30',
        'Icp/MaxCorrespondenceDistance': '0.1',

        # 2D occupancy grid (replaces slam_toolbox)
        'Grid/Sensor':        '0',          # use /scan
        'Grid/RangeMax':      '8.0',
        'Grid/CellSize':      '0.05',

        # 3D voxel cloud
        'cloud_voxel_size':   0.05,

        # Pi 4 memory budget
        'Mem/STMSize':        '30',
        'Rtabmap/DetectionRate': '1',
        'Rtabmap/TimeThr':    '700',

        # Save path
        'database_path':      os.path.expanduser('~/cave_map.db'),

        'use_sim_time':       False,
    }]

    rtabmap_remappings = [
        ('scan',        '/scan'),
        ('scan_cloud',  '/scan_tilted'),   # tilted LiDAR on /scan_tilted
        ('odom',        '/odom'),
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',   default_value='false'),
        DeclareLaunchArgument('localization',   default_value='false'),

        # ── RTAB-Map core node ────────────────────────────────────────────
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=rtabmap_parameters,
            remappings=rtabmap_remappings,
            arguments=['--delete_db_on_start'],  # fresh map each run
        ),

        # ── Visualisation helper (optional, uncomment if needed) ──────────
        # Node(
        #     package='rtabmap_viz',
        #     executable='rtabmap_viz',
        #     output='screen',
        #     parameters=rtabmap_parameters,
        #     remappings=rtabmap_remappings,
        # ),
    ])
