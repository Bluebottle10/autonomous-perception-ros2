"""
Semantic PointCloud Fusion Node
================================
Fuses a depth image with a segmentation mask to produce a colored 3D point cloud.

This node projects each pixel from the 2D camera view into 3D space using the
depth image, then colors each 3D point using the semantic segmentation mask.
The result is a PointCloud2 message where each point has XYZ position and RGB color.

Pipeline:
    /segmentation/color_mask (Image) ──┐
                                        ├──→ [Depth Unpack + Project] → /fusion/semantic_cloud (PointCloud2)
    /camera/depth (Image) ─────────────┘

Dependencies:
    - sensor_msgs_py (PointCloud2 creation)
    - OpenCV (image resize)
    - NumPy (3D projection math)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

import numpy as np
import cv2
import traceback


class PointCloudFusionNode(Node):
    """
    ROS2 node that creates semantic 3D point clouds from depth + segmentation.

    Parameters (ROS2 declared):
        segmentation_topic (str): Input segmentation mask topic.
        depth_topic (str): Input depth image topic.
        output_topic (str): Output PointCloud2 topic.
        camera_fov_deg (float): Camera field of view in degrees.
        far_plane_m (float): Unity camera far plane distance.
        frame_id (str): TF frame for the published point cloud.
        update_rate_hz (float): Processing rate.
    """

    def __init__(self):
        super().__init__("pointcloud_fusion_node")

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter("segmentation_topic", "/segmentation/color_mask")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("output_topic", "/fusion/semantic_cloud")
        self.declare_parameter("camera_fov_deg", 60.0)
        self.declare_parameter("far_plane_m", 1000.0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("update_rate_hz", 5.0)

        seg_topic = self.get_parameter("segmentation_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.fov_deg: float = self.get_parameter("camera_fov_deg").value
        self.far_plane: float = self.get_parameter("far_plane_m").value
        self.frame_id: str = self.get_parameter("frame_id").value
        update_hz: float = self.get_parameter("update_rate_hz").value

        # ── ROS2 Pub/Sub ─────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.bridge = CvBridge()

        self.create_subscription(Image, seg_topic, self._seg_callback, qos)
        self.create_subscription(Image, depth_topic, self._depth_callback, qos)
        self.pcl_pub = self.create_publisher(PointCloud2, output_topic, 10)

        # ── Internal state ───────────────────────────────────────────────
        self.latest_seg = None
        self.latest_depth = None
        self.img_w = 0
        self.img_h = 0
        self.x_norm = None
        self.y_norm = None

        self.create_timer(1.0 / update_hz, self._process_fusion)
        self.get_logger().info(f"PointCloud Fusion Node started | Output: {output_topic}")

    # ── Callbacks ────────────────────────────────────────────────────────
    def _seg_callback(self, msg: Image) -> None:
        self.latest_seg = msg

    def _depth_callback(self, msg: Image) -> None:
        self.latest_depth = msg

    # ── Intrinsics Update ────────────────────────────────────────────────
    def _update_intrinsics(self, w: int, h: int) -> None:
        """Recompute camera intrinsics and pixel-direction lookup tables."""
        if w == self.img_w and h == self.img_h:
            return  # No change

        self.img_w, self.img_h = w, h
        f_y = (h / 2.0) / np.tan(np.deg2rad(self.fov_deg) / 2.0)
        f_x = f_y
        c_x, c_y = w / 2.0, h / 2.0

        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        self.x_norm = (u_grid - c_x) / f_x
        self.y_norm = (v_grid - c_y) / f_y

        self.get_logger().info(f"Updated intrinsics for {w}x{h}")

    # ── Main Processing ──────────────────────────────────────────────────
    def _process_fusion(self) -> None:
        """Fuse depth + segmentation into a colored PointCloud2."""
        if self.latest_seg is None or self.latest_depth is None:
            return

        try:
            h = self.latest_depth.height
            w = self.latest_depth.width

            if len(self.latest_depth.data) == 0:
                self.get_logger().error("Empty depth data received!")
                return

            self._update_intrinsics(w, h)

            # 1. Decode depth (Unity 3-channel, depth in R channel)
            depth_buf = np.frombuffer(self.latest_depth.data, dtype=np.uint8)
            depth_img = depth_buf.reshape(h, w, 3)
            z = (depth_img[:, :, 0].astype(np.float32) / 255.0) * self.far_plane

            # 2. Decode segmentation
            seg_img = self.bridge.imgmsg_to_cv2(self.latest_seg, "bgr8")
            if seg_img.shape[:2] != (h, w):
                seg_img = cv2.resize(seg_img, (w, h), interpolation=cv2.INTER_NEAREST)

            # 3. Filter out far-plane (sky/infinity)
            mask = z < (self.far_plane * 0.95)

            # 4. Project to 3D (camera convention → ROS convention)
            # Forward = X (depth), Left = Y, Up = Z
            x_3d = z
            y_3d = -(self.x_norm * z)
            z_3d = -(self.y_norm * z)

            x_flat = x_3d[mask]
            y_flat = y_3d[mask]
            z_flat = z_3d[mask]

            # 5. Pack RGB color from segmentation mask
            b = seg_img[:, :, 0][mask].astype(np.uint32)
            g = seg_img[:, :, 1][mask].astype(np.uint32)
            r = seg_img[:, :, 2][mask].astype(np.uint32)
            rgb_packed = (255 << 24) | (r << 16) | (g << 8) | b  # ARGB
            rgb_float = rgb_packed.view(np.float32)

            # 6. Assemble points array
            points = np.zeros((x_flat.size, 4), dtype=np.float32)
            points[:, 0] = x_flat
            points[:, 1] = y_flat
            points[:, 2] = z_flat
            points[:, 3] = rgb_float

            # 7. Publish PointCloud2
            header = self.latest_depth.header
            header.frame_id = self.frame_id

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            pc_msg = point_cloud2.create_cloud(header, fields, points)
            self.pcl_pub.publish(pc_msg)

            self.get_logger().info(
                f"Published {len(x_flat)} points", throttle_duration_sec=2.0
            )

        except Exception as e:
            self.get_logger().error(f"PointCloud fusion failed: {e}")
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
