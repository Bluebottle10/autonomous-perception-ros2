"""
Visual Fusion Node
===================
Blends the raw camera image with the segmentation mask for visualization,
and provides a simple ego-vehicle safety status check.

Pipeline:
    /camera/image_raw (Image) ─────────┐
                                        ├──→ [Blend + Safety Check] → /fusion/visual_overlay (Image)
    /segmentation/color_mask (Image) ──┘

The node uses message_filters.ApproximateTimeSynchronizer to ensure
both images are from approximately the same time frame before blending.

Dependencies:
    - message_filters (time synchronization)
    - OpenCV (image blending)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import message_filters
import cv2
import numpy as np


class VisualFusionNode(Node):
    """
    ROS2 node that overlays segmentation on the camera feed for visualization.

    Parameters (ROS2 declared):
        image_topic (str): Raw camera image topic.
        segmentation_topic (str): Segmentation mask topic.
        output_topic (str): Blended output topic.
        blend_alpha (float): Weight for original image (0.0 - 1.0).
        sync_slop (float): Time tolerance for syncing messages (seconds).
    """

    def __init__(self):
        super().__init__("visual_fusion_node")

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("segmentation_topic", "/segmentation/color_mask")
        self.declare_parameter("output_topic", "/fusion/visual_overlay")
        self.declare_parameter("blend_alpha", 0.7)
        self.declare_parameter("sync_slop", 0.1)

        img_topic = self.get_parameter("image_topic").value
        seg_topic = self.get_parameter("segmentation_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.alpha: float = self.get_parameter("blend_alpha").value
        sync_slop: float = self.get_parameter("sync_slop").value

        # ── Synchronized Subscribers ─────────────────────────────────────
        self.img_sub = message_filters.Subscriber(self, Image, img_topic)
        self.mask_sub = message_filters.Subscriber(self, Image, seg_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.mask_sub], queue_size=10, slop=sync_slop
        )
        self.ts.registerCallback(self._fusion_callback)

        # ── Publisher ────────────────────────────────────────────────────
        self.pub = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

        self.get_logger().info(
            f"Visual Fusion Node started | Blend: {self.alpha:.0%} image + "
            f"{1.0 - self.alpha:.0%} segmentation"
        )

    def _fusion_callback(self, img_msg: Image, mask_msg: Image) -> None:
        """Blend raw image with segmentation mask and add safety status overlay."""
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, "bgr8")

        # Resize mask to match image if needed
        if cv_img.shape[:2] != cv_mask.shape[:2]:
            cv_mask = cv2.resize(cv_mask, (cv_img.shape[1], cv_img.shape[0]))

        # Weighted blend
        fused = cv2.addWeighted(cv_img, self.alpha, cv_mask, 1.0 - self.alpha, 0)

        # Safety status: check what's directly ahead (bottom-center of frame)
        h, w, _ = fused.shape
        center_pixel = cv_mask[h - 50, w // 2]  # Just in front of robot
        b, g, r = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])

        if r > 100 and b > 100 and g < 100:
            status, color = "ON ROAD", (0, 255, 0)
        elif r > 200 and b > 200:
            status, color = "ON SIDEWALK", (0, 165, 255)
        else:
            status, color = "OFF ROAD", (0, 0, 255)

        cv2.putText(
            fused, f"STATUS: {status}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

        # Publish
        out_msg = self.bridge.cv2_to_imgmsg(fused, encoding="bgr8")
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
