"""
Semantic Driver Node
=====================
Simple proportional controller that drives a robot based on the semantic costmap.

This node reads the OccupancyGrid, scans forward to find the road center,
and publishes velocity commands to steer the robot toward free space.

Pipeline:
    /costmap/semantic_map (OccupancyGrid) --> [Scan + P-Controller] --> /cmd_vel (Twist)

Control Logic:
    1. Scan the costmap column-by-column starting 1m ahead
    2. Find the first column with road data (value = 0)
    3. Compute the road center from free-space pixel indices
    4. Apply proportional steering: angular_z = Kp * (road_center - map_center)
    5. Stop if only obstacles are visible (no free space)

Dependencies:
    - nav_msgs (OccupancyGrid input)
    - geometry_msgs (Twist output)
    - NumPy (grid math)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist

import numpy as np


class SemanticDriverNode(Node):
    """
    ROS2 node that drives a robot by following free space in a semantic costmap.

    Parameters (ROS2 declared):
        costmap_topic (str): Input OccupancyGrid topic.
        cmd_vel_topic (str): Output velocity command topic.
        linear_speed (float): Forward speed in m/s.
        steering_kp (float): Proportional gain for steering.
        max_steering (float): Maximum angular velocity (rad/s).
    """

    def __init__(self):
        super().__init__("semantic_driver_node")

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter("costmap_topic", "/costmap/semantic_map")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.3)
        self.declare_parameter("steering_kp", 0.05)
        self.declare_parameter("max_steering", 1.0)

        costmap_topic = self.get_parameter("costmap_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value
        self.linear_speed: float = self.get_parameter("linear_speed").value
        self.kp: float = self.get_parameter("steering_kp").value
        self.max_steer: float = self.get_parameter("max_steering").value

        # ── ROS2 Pub/Sub ─────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, costmap_topic, self._map_callback, 1)
        self.pub = self.create_publisher(Twist, cmd_topic, 10)

        self.get_logger().info(
            f"Semantic Driver started | Speed: {self.linear_speed} m/s | Kp: {self.kp}"
        )

    def _map_callback(self, msg: OccupancyGrid) -> None:
        """Process incoming costmap and publish velocity commands."""
        w = msg.info.width    # X-axis (forward)
        h = msg.info.height   # Y-axis (left/right)
        res = msg.info.resolution

        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)

        # Scan forward: find first column with road or obstacle data
        start_col = int(1.0 / res)  # Skip 1m ahead (avoid bumper noise)
        found_road = False
        scan_slice = None

        for col in range(start_col, w, 2):
            col_data = grid[:, col]
            if np.any(col_data == 0) or np.any(col_data == 100):
                scan_slice = col_data
                found_road = True
                break

        cmd = Twist()

        if found_road and scan_slice is not None:
            # Find free-space indices in this column
            free_indices = np.where(scan_slice == 0)[0]

            if len(free_indices) > 0:
                # Steer toward the center of free space
                road_center = np.mean(free_indices)
                map_center = h / 2.0
                error = road_center - map_center

                steering = np.clip(error * self.kp, -self.max_steer, self.max_steer)

                cmd.linear.x = self.linear_speed
                cmd.angular.z = float(steering)
            else:
                # Only obstacles visible — stop
                self.get_logger().warn("Obstacle ahead! Stopping.", throttle_duration_sec=2.0)
        else:
            self.get_logger().warn("No map data. Waiting...", throttle_duration_sec=2.0)

        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticDriverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
