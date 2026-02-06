"""
PIDNet Semantic Segmentation Node (TensorRT)
=============================================
Real-time pixel-wise semantic segmentation using PIDNet accelerated with TensorRT.

PIDNet (Proportional-Integral-Derivative Network) is a real-time segmentation
architecture. This node runs the PIDNet-Large model, trained on the Cityscapes
dataset (19 urban scene classes), using TensorRT for GPU acceleration.

Pipeline:
    /camera/image_raw (Image) --> [Resize] --> [Normalize] --> [TensorRT] --> [Argmax + Colorize] --> /segmentation/color_mask (Image)

Cityscapes Classes (19):
    0: Road, 1: Sidewalk, 2: Building, 3: Wall, 4: Fence, 5: Pole,
    6: Traffic Light, 7: Traffic Sign, 8: Vegetation, 9: Terrain, 10: Sky,
    11: Person, 12: Rider, 13: Car, 14: Truck, 15: Bus, 16: Train,
    17: Motorcycle, 18: Bicycle

Dependencies:
    - TensorRT (with a pre-built .engine file)
    - CuPy (GPU memory management)
    - OpenCV (image resize)
    - cv_bridge (ROS <-> OpenCV conversion)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tensorrt as trt
import cupy as cp
import numpy as np
import cv2


# Cityscapes color palette: maps class ID → RGB color
CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128],   #  0: Road (Purple)
        [244, 35, 232],   #  1: Sidewalk (Pink)
        [70, 70, 70],     #  2: Building (Gray)
        [102, 102, 156],  #  3: Wall
        [190, 153, 153],  #  4: Fence
        [153, 153, 153],  #  5: Pole
        [250, 170, 30],   #  6: Traffic Light
        [220, 220, 0],    #  7: Traffic Sign
        [107, 142, 35],   #  8: Vegetation (Green)
        [152, 251, 152],  #  9: Terrain
        [70, 130, 180],   # 10: Sky (Blue)
        [220, 20, 60],    # 11: Person (Red)
        [255, 0, 0],      # 12: Rider
        [0, 0, 142],      # 13: Car (Dark Blue)
        [0, 0, 70],       # 14: Truck
        [0, 60, 100],     # 15: Bus
        [0, 80, 100],     # 16: Train
        [0, 0, 230],      # 17: Motorcycle
        [119, 11, 32],    # 18: Bicycle
    ],
    dtype=np.uint8,
)


class PIDNetTensorRTNode(Node):
    """
    ROS2 node for real-time semantic segmentation using PIDNet + TensorRT.

    Subscribes to a raw camera image, runs pixel-wise classification on the GPU,
    and publishes a colorized segmentation mask using the Cityscapes palette.

    Parameters (ROS2 declared):
        engine_path (str): Path to the PIDNet TensorRT .engine file.
        input_topic (str): Topic name for the input camera image.
        output_topic (str): Topic name for the colorized segmentation mask.
        model_width (int): Model input width (must match engine).
        model_height (int): Model input height (must match engine).
    """

    def __init__(self):
        super().__init__("pidnet_tensorrt_node")

        # ── Declare ROS2 Parameters ──────────────────────────────────────
        self.declare_parameter("engine_path", "pidnet_l.engine")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/segmentation/color_mask")
        self.declare_parameter("model_width", 1024)
        self.declare_parameter("model_height", 512)

        engine_path: str = self.get_parameter("engine_path").value
        input_topic: str = self.get_parameter("input_topic").value
        output_topic: str = self.get_parameter("output_topic").value
        self.model_w: int = self.get_parameter("model_width").value
        self.model_h: int = self.get_parameter("model_height").value

        # ── ROS2 Pub/Sub ─────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data
        )
        self.mask_publisher = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

        # ── ImageNet normalization constants ──────────────────────────────
        # PIDNet was trained with ImageNet preprocessing
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32)

        # ── TensorRT Engine Setup ────────────────────────────────────────
        self.get_logger().info(f"Loading TensorRT engine: {engine_path}")
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            self.get_logger().fatal(
                f"Engine file '{engine_path}' not found! "
                "Build it with: trtexec --onnx=pidnet_l.onnx --saveEngine=pidnet_l.engine"
            )
            raise SystemExit(1)

        self.trt_context = self.engine.create_execution_context()
        self._allocate_buffers()

        self.get_logger().info(
            f"PIDNet TensorRT Node started | "
            f"Model: {self.model_w}x{self.model_h} | "
            f"Input: {input_topic} | Output: {output_topic}"
        )

    # ── GPU Memory Allocation ────────────────────────────────────────────
    def _allocate_buffers(self) -> None:
        """Allocate persistent GPU buffers for TensorRT input/output tensors."""
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        self.get_logger().info(
            f"Input shape: {list(self.input_shape)} | "
            f"Output shape: {list(self.output_shape)}"
        )

        self.d_input = cp.empty(self.input_shape, dtype=cp.float32)
        self.d_output = cp.empty(self.output_shape, dtype=cp.float32)

        self.trt_context.set_tensor_address(self.input_name, self.d_input.data.ptr)
        self.trt_context.set_tensor_address(self.output_name, self.d_output.data.ptr)

    # ── Preprocessing ────────────────────────────────────────────────────
    def _preprocess(self, image_bgr: np.ndarray) -> cp.ndarray:
        """
        Prepare a BGR image for PIDNet inference.

        Steps:
            1. Resize to model input dimensions (1024x512)
            2. Upload to GPU (NumPy → CuPy)
            3. Normalize with ImageNet mean/std
            4. Transpose HWC → CHW
            5. Add batch dimension → (1, 3, H, W)

        Args:
            image_bgr: OpenCV BGR image, any resolution.

        Returns:
            CuPy array of shape (1, 3, model_h, model_w), float32.
        """
        resized = cv2.resize(image_bgr, (self.model_w, self.model_h))

        x = cp.asarray(resized, dtype=cp.float32)
        x /= 255.0
        x = (x - self.mean) / self.std  # ImageNet normalization

        x = cp.transpose(x, (2, 0, 1))  # HWC → CHW
        x = cp.expand_dims(x, axis=0)    # Add batch dim
        return x

    # ── ROS Callback ─────────────────────────────────────────────────────
    def image_callback(self, msg: Image) -> None:
        """
        Main callback: receives a ROS Image, runs segmentation, publishes colorized mask.

        Post-processing:
            - Output shape: (1, 19, H, W) — 19 Cityscapes class logits per pixel
            - Argmax across classes → (H, W) class map
            - Fancy indexing with CITYSCAPES_PALETTE → (H, W, 3) color image
        """
        try:
            # 1. ROS Image → OpenCV BGR
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 2. Preprocess and copy to GPU buffer
            input_data = self._preprocess(cv_img)
            self.d_input[:] = input_data

            # 3. Run TensorRT inference
            self.trt_context.execute_v2(
                bindings=[self.d_input.data.ptr, self.d_output.data.ptr]
            )

            # 4. Download result to CPU: shape (1, 19, H, W)
            raw_output = self.d_output.get()

            # 5. Argmax: find highest-scoring class per pixel → (H, W)
            class_map = np.argmax(np.squeeze(raw_output), axis=0).astype(np.uint8)

            # 6. Colorize using Cityscapes palette (fancy indexing)
            color_mask = CITYSCAPES_PALETTE[class_map]

            # 7. Publish with original timestamp for downstream synchronization
            out_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding="bgr8")
            out_msg.header = msg.header  # Preserve timestamp & frame_id
            self.mask_publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PIDNetTensorRTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
