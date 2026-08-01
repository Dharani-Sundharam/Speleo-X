#!/usr/bin/env python3
"""
auto_explorer.py — Autonomous Frontier Explorer for Cave Bot

Scans the SLAM /map for 'frontiers' (known free space pixels touching unknown space),
and continuously sends Nav2 goals to explore those frontiers until the entire cave is mapped.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
import random
import time

class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 1)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.map_msg = None
        self.goal_active = False
        self._last_attempt = 0
        
        self.get_logger().info("🤖 Autonomous Cave Explorer started! Waiting for /map...")

    def map_cb(self, msg):
        self.map_msg = msg
        # If we don't have an active goal and we haven't just failed one entirely too recently
        if not self.goal_active and (time.time() - self._last_attempt > 2.0):
            self.find_and_send_goal()

    def find_and_send_goal(self):
        if not self.map_msg:
            return
            
        w, h = self.map_msg.info.width, self.map_msg.info.height
        res = self.map_msg.info.resolution
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        data = self.map_msg.data

        frontiers = []
        
        # Scan map: step by 3 pixels to save CPU (≈ 15cm resolution scan)
        for y in range(1, h-1, 3):
            for x in range(1, w-1, 3):
                idx = y * w + x
                if data[idx] == 0:  # If cell is known free space
                    # Check immediate neighbours for unknown (-1)
                    neighbors = [
                        data[(y-1)*w + x], data[(y+1)*w + x],
                        data[y*w + (x-1)], data[y*w + (x+1)]
                    ]
                    if -1 in neighbors:
                        frontiers.append((x, y))
        if not frontiers:
            self.get_logger().info("🎉 No frontiers found! Map is 100% complete.")
            return

        # Pick a random frontier
        fx, fy = random.choice(frontiers)
        
        goal_x = ox + fx * res
        goal_y = oy + fy * res

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_x)
        goal_msg.pose.pose.position.y = float(goal_y)
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"📍 New Frontier Goal: x={goal_x:.2f}, y={goal_y:.2f}")
        
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Nav2 action server not available!")
            return
            
        self.goal_active = True
        self._last_attempt = time.time()
        
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("❌ Goal rejected by Nav2. Retrying another frontier...")
            self.goal_active = False
            return
            
        self.get_logger().info("✅ Goal accepted! Driving...")
        res_future = goal_handle.get_result_async()
        res_future.add_done_callback(self.goal_result_cb)

    def goal_result_cb(self, future):
        status = future.result().status
        if status == 4: # STATUS_SUCCEEDED
            self.get_logger().info("🏁 Reached frontier!")
        else:
            self.get_logger().warning(f"⚠️ Goal aborted or failed (status {status}). Picking a new one.")
            
        self.goal_active = False
        self._last_attempt = time.time()

def main():
    rclpy.init()
    node = CaveExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
