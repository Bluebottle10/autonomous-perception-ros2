"""
LiDAR PointCloud Listener Node
================================
Subscribes to simulated LiDAR point clouds from Unity and logs statistics.

This is a utility/debug node useful for verifying that the Unity LiDAR
simulation is publishing correctly.

Pipeline:
    /scan_cloud (PointCloud2) --> [Decode + Log]

Dependencies:
    - sensor_msgs_py (PointCloud2 decoding)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class LidarListenerNode(Node):
    """
    ROS2 node that subscribes to LiDAR point clouds and logs statistics.

    Parameters (ROS2 declared):
        lidar_topic (str): PointCloud2 input topic.
    """

    def __init__(self):
        super().__init__("lidar_listener_node")

        self.declare_parameter("lidar_topic", "/scan_cloud")
        lidar_topic = self.get_parameter("lidar_topic").value

        self.create_subscription(PointCloud2, lidar_topic, self._cloud_callback, 10)
        self.get_logger().info(f"LiDAR Listener started | Topic: {lidar_topic}")

    def _cloud_callback(self, msg: PointCloud2) -> None:
        """Decode incoming point cloud and log statistics."""
        points = list(
            pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        )

        count = len(points)
        if count > 0:
            first = points[0]
            self.get_logger().info(
                f"Cloud: {count} pts | First: ({first[0]:.2f}, {first[1]:.2f}, {first[2]:.2f})",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().warn("Received empty point cloud")


def main(args=None):
    rclpy.init(args=args)
    node = LidarListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
