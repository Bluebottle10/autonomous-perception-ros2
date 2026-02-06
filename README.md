# Autonomous Perception ROS2

Real-time perception pipeline for autonomous robot/drone navigation using ROS2 Jazzy on Ubuntu 24.04.

This workspace provides GPU-accelerated object detection (YOLOv11) and semantic segmentation (PIDNet), fused into a semantic costmap for autonomous navigation. Designed for use with [Unity simulation](https://github.com/Bluebottle10/autonomous-perception-unity) via the Unity Robotics Hub TCP Connector.

## Architecture

```
Unity Simulation (Windows)
    │
    ├── /camera/image_raw ──────────┐
    ├── /camera/depth ──────────────┤
    ├── /odom ──────────────────────┤
    ├── /tf ────────────────────────┤     ROS TCP
    └── /scan_cloud ────────────────┤──── Connector ────┐
                                    │                    │
                                    ▼                    ▼
                            ROS2 Jazzy (WSL2 / Ubuntu 24.04)
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌──────────────┐  ┌──────────────────────┐  ┌───────────────────────┐
│  YOLO v11    │  │  PIDNet Semantic     │  │  LiDAR Listener       │
│  TensorRT    │  │  Segmentation        │  │  (PointCloud2)        │
│  Detection   │  │  (TensorRT)          │  │                       │
└──────┬───────┘  └──────────┬───────────┘  └───────────────────────┘
       │                     │
       │    /yolo/detections │    /segmentation/color_mask
       │                     │
       │         ┌───────────┴───────────┐
       │         │                       │
       │         ▼                       ▼
       │  ┌──────────────┐    ┌──────────────────┐
       │  │  Semantic    │    │  PointCloud       │
       │  │  Costmap     │    │  Fusion           │
       │  │  Generation  │    │  (Depth + Seg)    │
       │  └──────┬───────┘    └──────────────────┘
       │         │
       │         │    /costmap/semantic_map
       │         ▼
       │  ┌──────────────┐
       │  │  Semantic    │──── /cmd_vel ────► Unity Vehicle
       │  │  Driver      │
       │  └──────────────┘
       │
       └──► /yolo/stop_signal ────► Unity Spline Animator
```

## Packages

| Package | Description | Key Nodes |
|---------|-------------|-----------|
| `perception_yolo` | YOLOv11 object detection with TensorRT | `yolo_tensorrt_node` |
| `perception_segmentation` | PIDNet semantic segmentation with TensorRT | `pidnet_tensorrt_node` |
| `perception_costmap` | Bird's-eye-view semantic costmap from depth + segmentation | `semantic_costmap_node` |
| `perception_fusion` | PointCloud fusion and visual overlay | `pointcloud_fusion_node`, `visual_fusion_node` |
| `perception_navigation` | Semantic driver, TF broadcaster, LiDAR listener | `semantic_driver_node`, `tf_broadcaster_node`, `lidar_listener_node` |
| `perception_bringup` | Launch files for full pipeline | `full_pipeline.launch.py`, `perception_only.launch.py` |

## Prerequisites

- **OS:** Ubuntu 24.04 (native or WSL2)
- **ROS2:** Jazzy Jalisco
- **GPU:** NVIDIA GPU with CUDA support
- **TensorRT:** 8.x or 10.x (for inference acceleration)
- **Python packages:** `cupy`, `opencv-python`, `numpy`
- **Unity:** [autonomous-perception-unity](https://github.com/Bluebottle10/autonomous-perception-unity) project (for simulation)

## Installation

```bash
# 1. Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/Bluebottle10/autonomous-perception-ros2.git autonomous_perception

# 2. Install ROS2 dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 3. Install Python dependencies
pip install cupy-cuda12x opencv-python numpy

# 4. Build TensorRT engine files (one-time setup)
# YOLO v11:
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine
# PIDNet:
trtexec --onnx=pidnet_l.onnx --saveEngine=pidnet_l.engine

# 5. Build the workspace
cd ~/ros2_ws
colcon build --packages-select perception_yolo perception_segmentation perception_costmap perception_fusion perception_navigation perception_bringup
source install/setup.bash
```

## Usage

### Launch the full pipeline
```bash
ros2 launch perception_bringup full_pipeline.launch.py
```

### Launch perception only (no navigation)
```bash
ros2 launch perception_bringup perception_only.launch.py
```

### Launch individual nodes
```bash
ros2 launch perception_yolo yolo.launch.py
ros2 launch perception_segmentation segmentation.launch.py
ros2 launch perception_costmap costmap.launch.py
ros2 launch perception_fusion fusion.launch.py
ros2 launch perception_navigation navigation.launch.py
```

### Custom parameters
```bash
ros2 launch perception_yolo yolo.launch.py params_file:=/path/to/custom_yolo.yaml
```

## Topic Reference

### Unity → ROS2 (Input)
| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | RGB camera feed |
| `/camera/depth` | `sensor_msgs/Image` | Depth image (encoded in R channel) |
| `/odom` | `nav_msgs/Odometry` | Robot odometry |
| `/tf` | `tf2_msgs/TFMessage` | Coordinate transforms |
| `/scan_cloud` | `sensor_msgs/PointCloud2` | LiDAR point cloud |

### ROS2 → Unity (Output)
| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands for robot |
| `/yolo/stop_signal` | `std_msgs/Bool` | Person detection stop signal |

### Internal ROS2 Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/detections` | `sensor_msgs/Image` | Annotated YOLO detections |
| `/segmentation/color_mask` | `sensor_msgs/Image` | Cityscapes color segmentation |
| `/costmap/semantic_map` | `nav_msgs/OccupancyGrid` | Navigation costmap |
| `/fusion/semantic_cloud` | `sensor_msgs/PointCloud2` | Colored 3D point cloud |
| `/fusion/visual_overlay` | `sensor_msgs/Image` | Blended camera + segmentation |

## Configuration

All parameters are configurable via YAML files in each package's `config/` directory. Key parameters:

### YOLO Detection (`perception_yolo/config/yolo_params.yaml`)
- `engine_path`: Path to TensorRT `.engine` file
- `confidence_threshold`: Detection confidence (default: 0.5)
- `nms_threshold`: Non-Maximum Suppression IoU threshold (default: 0.45)

### Semantic Segmentation (`perception_segmentation/config/segmentation_params.yaml`)
- `engine_path`: Path to PIDNet TensorRT `.engine` file
- `model_width` / `model_height`: Input dimensions (default: 1024x512)

### Costmap (`perception_costmap/config/costmap_params.yaml`)
- `grid_resolution`: Meters per cell (default: 0.2)
- `grid_width` / `grid_height`: Grid size in cells (default: 400x400)
- `safety_bubble_m`: Clear zone around robot (default: 2.0m)

## YouTube Tutorial Series

This project is part of an educational YouTube series on building autonomous perception systems:

1. **Video 1:** Setting Up ROS2 Jazzy + TensorRT for Autonomous Perception
2. **Video 2:** Implementing Real-Time Perception: YOLO v11 + Semantic Segmentation
3. **Video 3:** From Pixels to Navigation: Semantic Costmap Pipeline

## Related Repository

- **Unity Simulation:** [autonomous-perception-unity](https://github.com/Bluebottle10/autonomous-perception-unity) - Unity 6 sensor simulation project with LiDAR, radar, camera, and ROS2 integration.

## License

MIT License
