"""
YOLOv11 TensorRT Object Detection Node
=======================================
Real-time object detection using YOLOv11 accelerated with NVIDIA TensorRT.

This node subscribes to a camera image topic, runs inference on the GPU,
and publishes an annotated image with bounding boxes and class labels.

Pipeline:
    /camera/image_raw (Image) --> [Preprocess] --> [TensorRT Inference] --> [NMS] --> /yolo/detections (Image)

Dependencies:
    - TensorRT (with a pre-built .engine file from trtexec)
    - CuPy (for GPU memory management)
    - OpenCV (for drawing and NMS)
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


# COCO class names for labeling detections
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class YOLOv11TensorRTNode(Node):
    """
    ROS2 node that performs real-time object detection using YOLOv11 + TensorRT.

    Subscribes to a raw camera image, runs GPU-accelerated inference,
    applies Non-Maximum Suppression (NMS), and publishes the annotated result.

    Parameters (ROS2 declared):
        engine_path (str): Path to the TensorRT .engine file.
        confidence_threshold (float): Minimum detection confidence (0.0 - 1.0).
        nms_threshold (float): IoU threshold for Non-Maximum Suppression.
        input_topic (str): Topic name for the input camera image.
        output_topic (str): Topic name for the annotated output image.
    """

    def __init__(self):
        super().__init__("yolov11_tensorrt_node")

        # ── Declare ROS2 Parameters ──────────────────────────────────────
        self.declare_parameter("engine_path", "yolo11n.engine")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/yolo/detections")

        # Read parameter values
        engine_path: str = self.get_parameter("engine_path").value
        self.conf_thresh: float = self.get_parameter("confidence_threshold").value
        self.nms_thresh: float = self.get_parameter("nms_threshold").value
        input_topic: str = self.get_parameter("input_topic").value
        output_topic: str = self.get_parameter("output_topic").value

        # ── ROS2 Pub/Sub ─────────────────────────────────────────────────
        self.subscription = self.create_subscription(
            Image, input_topic, self.image_callback, qos_profile_sensor_data
        )
        self.debug_publisher = self.create_publisher(Image, output_topic, 10)
        self.bridge = CvBridge()

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
                "Build it with: trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine"
            )
            raise SystemExit(1)

        self.trt_context = self.engine.create_execution_context()
        self._allocate_buffers()

        self.get_logger().info(
            f"YOLOv11 TensorRT Node started | "
            f"Input: {input_topic} | Output: {output_topic} | "
            f"Confidence: {self.conf_thresh} | NMS: {self.nms_thresh}"
        )

    # ── GPU Memory Allocation ────────────────────────────────────────────
    def _allocate_buffers(self) -> None:
        """Allocate persistent GPU buffers for TensorRT input/output tensors."""
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)

        self.get_logger().info(f"Input shape: {input_shape} | Output shape: {output_shape}")

        # Pre-allocate GPU arrays (CuPy) — reused every frame
        self.d_input = cp.empty(input_shape, dtype=cp.float32)
        self.d_output = cp.empty(output_shape, dtype=cp.float32)

        # Bind memory addresses to TensorRT execution context
        self.trt_context.set_tensor_address(self.input_name, self.d_input.data.ptr)
        self.trt_context.set_tensor_address(self.output_name, self.d_output.data.ptr)

    # ── Preprocessing (CPU → GPU) ────────────────────────────────────────
    def _preprocess(self, image_bgr: np.ndarray) -> cp.ndarray:
        """
        Prepare a BGR image for YOLO inference.

        Steps:
            1. Upload to GPU (NumPy → CuPy)
            2. Convert to float32 and normalize to [0, 1]
            3. Transpose HWC → CHW
            4. Add batch dimension → (1, 3, H, W)

        Args:
            image_bgr: OpenCV BGR image (H, W, 3), uint8.

        Returns:
            CuPy array of shape (1, 3, H, W), float32.
        """
        x = cp.asarray(image_bgr, dtype=cp.float32)
        x /= 255.0
        x = cp.transpose(x, (2, 0, 1))  # HWC → CHW
        x = cp.expand_dims(x, axis=0)    # Add batch dim
        return x

    # ── Postprocessing (GPU → CPU → Draw) ────────────────────────────────
    def _postprocess(self, raw_output: np.ndarray, original_image: np.ndarray) -> None:
        """
        Decode YOLO output, apply NMS, draw boxes, and publish the result.

        YOLO output format: (1, 84, N) where 84 = 4 (bbox) + 80 (COCO classes)
        and N is the number of candidate detections.

        Args:
            raw_output: NumPy array from TensorRT, shape (1, 84, N).
            original_image: BGR image to draw detections on.
        """
        # Reshape: (1, 84, N) → (N, 84) where each row = [cx, cy, w, h, class_scores...]
        predictions = np.squeeze(raw_output).T

        # Filter by confidence threshold
        scores = np.max(predictions[:, 4:], axis=1)
        keep = scores > self.conf_thresh
        predictions = predictions[keep]
        scores = scores[keep]

        if len(scores) == 0:
            # No detections — publish original image unchanged
            self._publish_image(original_image)
            return

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Convert YOLO format (cx, cy, w, h) → OpenCV NMS format (x, y, w, h)
        boxes = []
        for row in predictions:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, scores.tolist(), self.conf_thresh, self.nms_thresh
        )

        # Draw bounding boxes
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"ID:{class_id}"

                # Draw box and label
                color = (0, 255, 0)  # Green
                cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    original_image,
                    f"{label}: {scores[i]:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        self._publish_image(original_image)

    # ── Publish Helper ───────────────────────────────────────────────────
    def _publish_image(self, image: np.ndarray) -> None:
        """Convert an OpenCV image to a ROS Image message and publish it."""
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.debug_publisher.publish(msg)

    # ── ROS Callback ─────────────────────────────────────────────────────
    def image_callback(self, msg: Image) -> None:
        """
        Main callback: receives a ROS Image, runs YOLO inference, publishes results.

        This is called every time a new image arrives on the input topic.
        """
        try:
            # 1. ROS Image → OpenCV BGR
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 2. Preprocess (upload to GPU, normalize, transpose)
            input_data = self._preprocess(cv_img)

            # 3. Copy into TensorRT's pre-allocated input buffer
            self.d_input[:] = input_data

            # 4. Run TensorRT inference
            self.trt_context.execute_v2(
                bindings=[self.d_input.data.ptr, self.d_output.data.ptr]
            )

            # 5. Download results from GPU → CPU
            raw_results = self.d_output.get()

            # 6. Postprocess (NMS, draw boxes, publish)
            self._postprocess(raw_results, cv_img)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv11TensorRTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
