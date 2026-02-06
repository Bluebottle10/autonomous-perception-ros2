"""
TF Broadcaster Node
====================
Publishes coordinate frame transforms for the robot and camera.

Transforms published:
    odom → base_link     (from odometry data)
    base_link → camera_link  (static offset: camera position on robot)

These transforms are essential for:
    - Connecting the costmap/pointcloud to the robot's physical frame
    - RViz visualization
    - Nav2 localization

Dependencies:
    - tf2_ros (transform broadcasting)
    - nav_msgs (Odometry subscription)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class TFBroadcasterNode(Node):
    """
    ROS2 node that broadcasts TF transforms from odometry.

    Parameters (ROS2 declared):
        odom_topic (str): Odometry input topic.
        camera_x_offset (float): Camera forward offset from base (meters).
        camera_z_offset (float): Camera height from base (meters).
    """

    def __init__(self):
        super().__init__("tf_broadcaster_node")

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("camera_x_offset", 0.5)
        self.declare_parameter("camera_z_offset", 1.5)

        odom_topic = self.get_parameter("odom_topic").value
        self.cam_x: float = self.get_parameter("camera_x_offset").value
        self.cam_z: float = self.get_parameter("camera_z_offset").value

        # ── TF Broadcaster ───────────────────────────────────────────────
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, odom_topic, self._handle_odom, 10)

        self.get_logger().info(
            f"TF Broadcaster started | Camera offset: x={self.cam_x}m, z={self.cam_z}m"
        )

    def _handle_odom(self, msg: Odometry) -> None:
        """Broadcast odom→base_link and base_link→camera_link transforms."""
        stamp = self.get_clock().now().to_msg()

        # 1. odom → base_link (from odometry)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = 0.0  # Force 2D
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)

        # 2. base_link → camera_link (static camera mount)
        t_cam = TransformStamped()
        t_cam.header.stamp = stamp
        t_cam.header.frame_id = "base_link"
        t_cam.child_frame_id = "camera_link"
        t_cam.transform.translation.x = self.cam_x
        t_cam.transform.translation.y = 0.0
        t_cam.transform.translation.z = self.cam_z
        t_cam.transform.rotation.w = 1.0  # Identity (no rotation)
        self.br.sendTransform(t_cam)


def main(args=None):
    rclpy.init(args=args)
    node = TFBroadcasterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
