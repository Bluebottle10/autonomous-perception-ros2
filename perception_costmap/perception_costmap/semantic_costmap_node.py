"""
Semantic Costmap Generation Node
=================================
Converts a segmentation mask + depth image into a 2D OccupancyGrid for navigation.

This node fuses the semantic class information (road vs obstacle) with depth data
to project the 2D camera view into a bird's-eye-view occupancy grid. The grid is
suitable for use with Nav2 or custom path planners.

Pipeline:
    /segmentation/color_mask (Image) ──┐
                                        ├──→ [Project to BEV] → /costmap/semantic_map (OccupancyGrid)
    /camera/depth (Image) ─────────────┘

Grid Encoding:
    -1  = Unknown (no data)
     0  = Free space (road)
    100 = Occupied (obstacle)

Dependencies:
    - OpenCV (image processing, morphological dilation)
    - NumPy (grid math, camera projection)
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

import numpy as np
import cv2


class SemanticCostmapNode(Node):
    """
    ROS2 node that generates a semantic occupancy grid from segmentation + depth.

    The node identifies road pixels (purple in Cityscapes palette) and treats
    everything else as obstacles. Depth data is used to project these labels
    into real-world coordinates on a 2D grid.

    Parameters (ROS2 declared):
        segmentation_topic (str): Input segmentation mask topic.
        depth_topic (str): Input depth image topic.
        output_topic (str): Output OccupancyGrid topic.
        camera_fov_deg (float): Camera field of view in degrees.
        camera_height_m (float): Camera height above ground in meters.
        max_obstacle_height_m (float): Ignore obstacles taller than this.
        grid_resolution (float): Meters per grid cell.
        grid_width (int): Grid width in cells.
        grid_height (int): Grid height in cells.
        offset_behind_m (float): Meters of map behind the robot.
        far_plane_m (float): Unity camera far plane distance in meters.
        update_rate_hz (float): Map publication rate.
        safety_bubble_m (float): Radius of clear zone around robot.
    """

    def __init__(self):
        super().__init__("semantic_costmap_node")

        # ── Declare Parameters ───────────────────────────────────────────
        self.declare_parameter("segmentation_topic", "/segmentation/color_mask")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("output_topic", "/costmap/semantic_map")
        self.declare_parameter("camera_fov_deg", 60.0)
        self.declare_parameter("camera_height_m", 1.5)
        self.declare_parameter("max_obstacle_height_m", 2.5)
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("grid_width", 400)
        self.declare_parameter("grid_height", 400)
        self.declare_parameter("offset_behind_m", 10.0)
        self.declare_parameter("far_plane_m", 100.0)
        self.declare_parameter("max_range_m", 500.0)
        self.declare_parameter("update_rate_hz", 5.0)
        self.declare_parameter("safety_bubble_m", 2.0)
        self.declare_parameter("road_color_tolerance", 40)

        # Read parameters
        seg_topic = self.get_parameter("segmentation_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.fov_deg: float = self.get_parameter("camera_fov_deg").value
        self.camera_height: float = self.get_parameter("camera_height_m").value
        self.max_obs_height: float = self.get_parameter("max_obstacle_height_m").value
        self.grid_res: float = self.get_parameter("grid_resolution").value
        self.grid_w: int = self.get_parameter("grid_width").value
        self.grid_h: int = self.get_parameter("grid_height").value
        self.offset_m: float = self.get_parameter("offset_behind_m").value
        self.far_plane: float = self.get_parameter("far_plane_m").value
        self.max_range: float = self.get_parameter("max_range_m").value
        update_hz: float = self.get_parameter("update_rate_hz").value
        self.bubble_m: float = self.get_parameter("safety_bubble_m").value
        self.road_tol: int = self.get_parameter("road_color_tolerance").value

        # ── Subscribers (Sensor QoS for Unity streams) ───────────────────
        self.create_subscription(Image, seg_topic, self._seg_callback, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._depth_callback, qos_profile_sensor_data)

        # ── Publisher (Transient Local for late-joining subscribers) ──────
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, output_topic, map_qos)

        self.bridge = CvBridge()
        self.latest_seg = None
        self.latest_depth = None

        # Timer-based map update
        self.create_timer(1.0 / update_hz, self._update_map)

        self.get_logger().info(
            f"Semantic Costmap Node started | "
            f"Grid: {self.grid_w}x{self.grid_h} @ {self.grid_res}m/cell | "
            f"Rate: {update_hz}Hz"
        )

    # ── Callbacks ────────────────────────────────────────────────────────
    def _seg_callback(self, msg: Image) -> None:
        self.latest_seg = msg

    def _depth_callback(self, msg: Image) -> None:
        self.latest_depth = msg

    # ── Main Processing ──────────────────────────────────────────────────
    def _update_map(self) -> None:
        """Generate and publish the occupancy grid from latest sensor data."""
        if self.latest_seg is None or self.latest_depth is None:
            return

        try:
            # ── 1. Decode depth image ────────────────────────────────────
            h, w = self.latest_depth.height, self.latest_depth.width

            # Compute camera intrinsics dynamically from image dimensions
            f_y = (h / 2.0) / np.tan(np.deg2rad(self.fov_deg) / 2.0)
            f_x = f_y  # Square pixels
            c_x = w / 2.0
            c_y = h / 2.0

            # Unity encodes depth in the R channel of a 3-channel image
            depth_buf = np.frombuffer(self.latest_depth.data, dtype=np.uint8)
            depth_raw = depth_buf.reshape(h, w, 3)
            z_meters = (depth_raw[:, :, 0].astype(np.float32) / 255.0) * self.far_plane

            # ── 2. Decode segmentation mask ──────────────────────────────
            seg_img = self.bridge.imgmsg_to_cv2(self.latest_seg, "bgr8")
            if seg_img.shape[:2] != (h, w):
                seg_img = cv2.resize(seg_img, (w, h), interpolation=cv2.INTER_NEAREST)

            # ── 3. Classify pixels: road vs obstacle ─────────────────────
            # Road = Cityscapes purple (R=128, G=64, B=128) with tolerance
            b, g, r = seg_img[:, :, 0], seg_img[:, :, 1], seg_img[:, :, 2]
            tol = self.road_tol
            is_road = (
                (np.abs(r.astype(int) - 128) < tol)
                & (np.abs(g.astype(int) - 64) < tol)
                & (np.abs(b.astype(int) - 128) < tol)
            )
            is_obstacle = ~is_road

            # ── 4. Height filtering ──────────────────────────────────────
            # Ignore obstacles above max_obstacle_height (e.g., tree canopy)
            v_grid, _ = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            y_real = ((v_grid - c_y) / f_y) * z_meters
            threshold_y = -(self.max_obs_height - self.camera_height)
            is_low_enough = y_real > threshold_y

            final_obstacle = is_obstacle & is_low_enough

            # ── 5. Project to bird's-eye-view grid ───────────────────────
            skip = 4  # Subsample for performance
            z_sub = z_meters[::skip, ::skip]
            obs_sub = final_obstacle[::skip, ::skip]
            road_sub = is_road[::skip, ::skip]

            u_sub = np.arange(0, w, skip)
            x_norm = (u_sub - c_x) / f_x

            # Camera frame → map frame
            map_x = z_sub                      # Forward
            map_y = -(x_norm[None, :] * z_sub)  # Left/Right

            # Convert to grid indices
            offset_cells = int(self.offset_m / self.grid_res)
            idx_x = (map_x / self.grid_res).astype(int) + offset_cells
            idx_y = (map_y / self.grid_res).astype(int) + (self.grid_w // 2)

            valid = (
                (idx_x >= 0) & (idx_x < self.grid_h)
                & (idx_y >= 0) & (idx_y < self.grid_w)
                & (z_sub > 0.5) & (z_sub < self.max_range)
            )

            # ── 6. Fill the occupancy grid ───────────────────────────────
            obs_grid = np.zeros((self.grid_w, self.grid_h), dtype=np.uint8)
            road_grid = np.zeros((self.grid_w, self.grid_h), dtype=np.uint8)

            obs_grid[idx_y[valid & obs_sub], idx_x[valid & obs_sub]] = 1
            road_grid[idx_y[valid & road_sub], idx_x[valid & road_sub]] = 1

            # Safety bubble: clear zone around the robot
            robot_gx = offset_cells
            robot_gy = self.grid_w // 2
            br = int(self.bubble_m / self.grid_res)
            obs_grid[
                max(0, robot_gy - br): min(self.grid_w, robot_gy + br),
                max(0, robot_gx - br): min(self.grid_h, robot_gx + br),
            ] = 0

            # Dilate obstacles to fill gaps
            kernel = np.ones((3, 3), np.uint8)
            obs_dilated = cv2.dilate(obs_grid, kernel, iterations=1)

            # ── 7. Compose final map ─────────────────────────────────────
            final_map = np.full((self.grid_w, self.grid_h), -1, dtype=np.int8)
            final_map[road_grid > 0] = 0       # Free space
            final_map[obs_dilated > 0] = 100    # Obstacles override road

            # ── 8. Publish OccupancyGrid ─────────────────────────────────
            msg = OccupancyGrid()
            now = time.time()
            msg.header.stamp.sec = int(now)
            msg.header.stamp.nanosec = int((now - int(now)) * 1e9)
            msg.header.frame_id = "map"

            msg.info.resolution = self.grid_res
            msg.info.width = self.grid_h
            msg.info.height = self.grid_w
            msg.info.origin.position.x = -self.offset_m
            msg.info.origin.position.y = -(self.grid_w * self.grid_res) / 2.0
            msg.info.origin.position.z = 0.0

            msg.data = final_map.flatten().tolist()
            self.map_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Costmap generation failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SemanticCostmapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
