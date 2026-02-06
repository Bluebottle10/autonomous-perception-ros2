# YouTube Video Series: Autonomous Perception with ROS2 + Unity

## Series Overview
A 3-part educational series teaching viewers how to build a real-time autonomous perception pipeline from scratch using ROS2 Jazzy, TensorRT, and Unity simulation.

**Target audience:** Robotics students, ML engineers getting into robotics, Unity developers curious about ROS2

**GitHub repos:**
- ROS2: https://github.com/Bluebottle10/autonomous-perception-ros2
- Unity: https://github.com/Bluebottle10/autonomous-perception-unity

---

## VIDEO 1: Setting Up ROS2 Jazzy + TensorRT for Autonomous Perception

**Duration:** ~20-25 minutes
**Thumbnail idea:** Split screen of Unity simulation on left, RViz/terminal on right, with "ROS2 + TensorRT + Unity" text overlay

### Hook (0:00 - 1:00)
- Show the final result: drone/robot navigating autonomously in Unity with live YOLO detections, segmentation overlay, and costmap visualization in RViz
- "In this 3-part series, I'll show you how to build this entire perception pipeline from scratch"

### Part 1: What We're Building (1:00 - 3:00)
- Architecture diagram walkthrough (use the README diagram)
- Explain the data flow: Unity → ROS TCP → YOLO/PIDNet → Costmap → Navigation
- Explain why TensorRT: real-time inference on GPU, 10-50x faster than PyTorch
- Quick overview of all 3 videos

### Part 2: Environment Setup (3:00 - 10:00)

#### 2a: WSL2 + Ubuntu 24.04 (3:00 - 5:00)
- Installing WSL2 on Windows
- Setting up Ubuntu 24.04
- Why WSL2: GPU passthrough, easy networking with Unity on Windows

#### 2b: ROS2 Jazzy Installation (5:00 - 7:00)
- `sudo apt install ros-jazzy-desktop`
- Source setup: `source /opt/ros/jazzy/setup.bash`
- Create workspace: `mkdir -p ~/ros2_ws/src`
- Verify: `ros2 topic list`

#### 2c: NVIDIA + TensorRT Setup (7:00 - 10:00)
- CUDA installation (WSL2 specific steps)
- TensorRT installation
- Verify GPU access: `nvidia-smi` from WSL2
- Install CuPy: `pip install cupy-cuda12x`
- Quick test: allocate GPU memory from Python

### Part 3: Unity Side Setup (10:00 - 15:00)

#### 3a: Unity Project Overview (10:00 - 12:00)
- Open the SensorResearch project
- Tour the key scripts: RosCameraPublisher, DepthCameraPublisher, OdometryPublisher
- Show the Inspector: topic names, resolutions, publish rates

#### 3b: ROS TCP Connector Setup (12:00 - 14:00)
- Package Manager → ROS TCP Connector
- Robotics → ROS Settings → set IP address
- Start the ROS TCP Endpoint on WSL2 side
- Demo: run a simple publisher, see the topic in `ros2 topic list`

#### 3c: Verify End-to-End Connection (14:00 - 15:00)
- Unity Play → camera image publishing
- `ros2 topic echo /camera/image_raw` shows data flowing
- rqt_image_view showing the Unity camera feed

### Part 4: Project Structure Deep Dive (15:00 - 20:00)
- Clone the autonomous-perception-ros2 repo
- Walk through each package: what it does, what files are in it
- Explain ROS2 package conventions: package.xml, setup.py, setup.cfg
- Show a config YAML: how parameters are externalized
- Show a launch file: how nodes are started with configs
- `colcon build` and `source install/setup.bash`

### Part 5: Building Your First TensorRT Engine (20:00 - 23:00)
- What is ONNX? What is a TensorRT engine?
- Export YOLO to ONNX: `yolo export model=yolo11n.pt format=onnx`
- Build engine: `trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine`
- Same for PIDNet: export and build
- Explain: engine files are GPU-specific (can't share between different GPUs)

### Outro (23:00 - 25:00)
- Recap: what we set up
- Preview of Video 2: "Next, we'll implement the YOLO and segmentation nodes"
- Like, subscribe, link to GitHub repos in description

---

## VIDEO 2: Real-Time Perception — YOLO v11 + Semantic Segmentation in ROS2

**Duration:** ~25-30 minutes
**Thumbnail idea:** Split image showing YOLO bounding boxes on one side, colorful segmentation mask on the other, "TensorRT Real-Time" badge

### Hook (0:00 - 1:00)
- Live demo: Unity scene running, YOLO detections and segmentation happening in real-time
- "Today we implement both of these perception nodes from scratch using TensorRT"

### Part 1: TensorRT Inference Explained (1:00 - 4:00)
- How TensorRT works: ONNX → Engine → GPU inference
- Key concepts: CUDA streams, GPU memory (CuPy), input/output bindings
- The inference loop: preprocess → copy to GPU → execute → copy from GPU → postprocess
- Why this is 10-50x faster than PyTorch

### Part 2: YOLO v11 TensorRT Node (4:00 - 15:00)

#### 2a: Node Structure (4:00 - 6:00)
- Open `perception_yolo/perception_yolo/yolo_tensorrt_node.py`
- Walk through the class structure: `__init__`, callbacks, processing
- ROS2 parameters: engine_path, confidence, NMS threshold, topics
- Subscriber (input image) and Publisher (annotated output)

#### 2b: GPU Memory Allocation (6:00 - 8:00)
- `_allocate_buffers()`: creating CuPy arrays for TensorRT I/O
- Explain tensor shapes: input `(1, 3, 960, 1280)`, output `(1, 84, 25200)`
- What the 84 means: 4 box coords + 80 COCO class scores
- What the 25200 means: total anchor predictions across scales

#### 2c: Preprocessing Pipeline (8:00 - 10:00)
- `_preprocess()`: resize → GPU upload → normalize → HWC→CHW → batch dim
- Show the actual data transformations with shapes at each step
- Why we normalize: model was trained with normalized inputs

#### 2d: Post-processing — NMS (10:00 - 13:00)
- Raw output: 25,200 predictions — most are garbage
- Confidence filtering: keep only predictions above threshold
- Non-Maximum Suppression: remove overlapping boxes
- COCO class names: mapping class IDs to human-readable labels
- Drawing boxes and labels on the image

#### 2e: Live Demo (13:00 - 15:00)
- Launch: `ros2 launch perception_yolo yolo.launch.py`
- Unity running with pedestrians and vehicles
- Show detections in rqt_image_view
- Tune parameters live: change confidence threshold, see the effect

### Part 3: PIDNet Semantic Segmentation Node (15:00 - 24:00)

#### 3a: What is Semantic Segmentation? (15:00 - 17:00)
- Pixel-wise classification vs object detection (bounding boxes)
- Cityscapes dataset: 19 urban classes
- Show the color palette: road=purple, car=dark blue, person=red, sky=blue
- Why PIDNet: fast, accurate, designed for real-time driving

#### 3b: Node Walkthrough (17:00 - 20:00)
- Open `perception_segmentation/perception_segmentation/pidnet_tensorrt_node.py`
- Input shape: `(1, 3, 512, 1024)` — different from YOLO!
- Output shape: `(1, 19, 512, 1024)` — 19 class scores per pixel
- Preprocessing: ImageNet normalization (mean/std), resize, HWC→CHW

#### 3c: Colorization (20:00 - 22:00)
- `np.argmax(output, axis=0)` → class map (H, W) with values 0-18
- Fancy indexing with color palette: `CITYSCAPES_PALETTE[class_map]`
- This is the magic one-liner that converts class IDs to colors
- Show the resulting segmentation mask

#### 3d: Live Demo (22:00 - 24:00)
- Launch: `ros2 launch perception_segmentation segmentation.launch.py`
- Unity driving scene with segmentation overlay
- Show in rqt_image_view: raw camera vs segmentation mask
- Run YOLO + Segmentation simultaneously — real-time on GPU

### Part 4: Running Both Together (24:00 - 27:00)
- Launch perception_only: `ros2 launch perception_bringup perception_only.launch.py`
- Both nodes processing the same camera feed in parallel
- Show visual fusion: blended overlay of camera + segmentation
- `ros2 topic hz` — verify real-time frame rates
- rqt_graph: visualize the node/topic connections

### Outro (27:00 - 30:00)
- Recap: built two GPU-accelerated perception nodes
- Preview Video 3: "Next, we turn these pixel-level results into a navigation map"
- Show a quick teaser of the costmap and autonomous driving
- GitHub links, subscribe

---

## VIDEO 3: From Pixels to Navigation — Semantic Costmap Pipeline

**Duration:** ~25-30 minutes
**Thumbnail idea:** Camera view transforming into bird's-eye costmap, with a robot path drawn on it, "Full Pipeline" badge

### Hook (0:00 - 1:30)
- Show the complete pipeline: camera feed → segmentation → costmap → robot driving autonomously
- "We've detected objects and segmented the scene. Now we turn that into a map the robot can actually navigate"

### Part 1: The Bridge — Depth + Segmentation Fusion (1:30 - 6:00)

#### 1a: Why Depth Matters (1:30 - 3:00)
- Segmentation tells us WHAT is in each pixel (road, car, person)
- Depth tells us WHERE it is in 3D space
- Together: "there's road 5 meters ahead, and a car 10 meters to the left"
- Show the depth image from Unity (grayscale encoding)

#### 1b: Camera Projection Math (3:00 - 5:00)
- Camera intrinsics: focal length, principal point, FOV
- Pixel (u, v) + depth (z) → 3D point (X, Y, Z)
- The projection equations (with visual diagram)
- Code walkthrough: `_update_intrinsics()` in pointcloud_fusion_node.py

#### 1c: PointCloud Fusion Node (5:00 - 6:00)
- Subscribe to segmentation + depth
- Project every pixel to 3D, color it with segmentation class
- Publish as PointCloud2 — viewable in RViz
- Demo: colored 3D point cloud in RViz

### Part 2: Semantic Costmap Generation (6:00 - 16:00)

#### 2a: What is a Costmap? (6:00 - 8:00)
- OccupancyGrid: 2D grid, each cell = free (0), occupied (100), unknown (-1)
- Bird's-eye-view (BEV) projection: camera view → top-down map
- Grid parameters: resolution (0.2m/cell), size (400x400 = 80m x 80m)
- This is what navigation planners (Nav2, RRT*, A*) need

#### 2b: Road Detection (8:00 - 10:00)
- Cityscapes road color: purple (128, 64, 128)
- Color matching with tolerance: why exact matching fails
- `is_road` mask: boolean array of road vs non-road pixels
- Height filtering: ignore obstacles above max height (tree canopy, bridges)

#### 2c: BEV Projection Code (10:00 - 13:00)
- Walk through `_update_map()` step by step
- Subsampling (every 4th pixel) for performance
- Camera frame → map frame coordinate conversion
- Grid index calculation: continuous (meters) → discrete (cells)
- Safety bubble: clear zone around the robot
- Obstacle dilation: fill gaps between sparse obstacle points

#### 2d: Publishing OccupancyGrid (13:00 - 14:00)
- ROS2 OccupancyGrid message structure
- Map origin, resolution, frame_id
- QoS: Transient Local for late-joining subscribers (RViz)

#### 2e: Live Demo (14:00 - 16:00)
- Launch costmap node alongside segmentation
- RViz: show the costmap updating in real-time
- Overlay on the camera view: point out road (green) vs obstacles (red)
- Change parameters live: grid resolution, safety bubble size

### Part 3: TF Transforms — The Glue (16:00 - 19:00)

#### 3a: Why TF Matters (16:00 - 17:00)
- Every sensor has its own frame: camera_link, lidar_link, base_link
- TF tree connects them all: odom → base_link → camera_link → lidar_link
- Without TF: the costmap and point cloud would be in the wrong place

#### 3b: Unity-Side TF Publishing (17:00 - 18:00)
- OdometryPublisher.cs: odom → base_link (dynamic)
- LidarStaticTfPublisher.cs: base_link → lidar_link (static)
- Coordinate conversion: Unity (Z-fwd, Y-up) → ROS (X-fwd, Z-up)

#### 3c: ROS-Side TF Broadcasting (18:00 - 19:00)
- tf_broadcaster_node.py: subscribes to /odom, broadcasts transforms
- Static camera transform: base_link → camera_link
- Show TF tree in RViz

### Part 4: Autonomous Navigation — Closing the Loop (19:00 - 25:00)

#### 4a: Semantic Driver (19:00 - 21:00)
- Simple proportional controller
- Reads the costmap, finds free space (road)
- Calculates road center, steers toward it
- Publishes /cmd_vel → Unity RosVehicleController subscribes

#### 4b: Full Pipeline Launch (21:00 - 23:00)
- `ros2 launch perception_bringup full_pipeline.launch.py`
- ALL nodes running: YOLO + PIDNet + Costmap + Fusion + Driver + TF
- Show rqt_graph: the complete node/topic network
- Show the robot driving itself in Unity

#### 4c: Tuning and Experimentation (23:00 - 25:00)
- Adjust steering Kp: too high = oscillation, too low = slow response
- Adjust safety bubble: bigger = more conservative
- Adjust confidence threshold: lower = more detections but more false positives
- Show failure cases and how to debug them

### Part 5: What's Next (25:00 - 28:00)
- Where to go from here:
  - Nav2 integration (replace simple driver with full navigation stack)
  - SLAM (build a persistent map)
  - Multi-sensor fusion (combine LiDAR + camera costmaps)
  - Path planning algorithms (RRT*, A*, DWB)
  - Behavior trees for mission-level planning
- Preview of future videos in the series

### Outro (28:00 - 30:00)
- Recap the entire 3-video journey
- Show the complete pipeline one more time
- GitHub repos (both), subscribe, community/Discord if applicable
- "Build something amazing with this — I'd love to see what you create"

---

## Production Notes

### Screen Recording Tips
- Use OBS Studio with multiple scenes:
  - Scene 1: Unity fullscreen
  - Scene 2: VSCode/terminal fullscreen
  - Scene 3: Split screen (Unity + RViz)
  - Scene 4: Picture-in-picture (code + demo)
- Record at 1080p 60fps minimum (4K if possible)
- Use a good microphone — audio quality matters more than video

### Graphics / Animations
- Architecture diagrams (use draw.io, Excalidraw, or Figma)
- Data flow animations showing messages moving between nodes
- Side-by-side comparisons: raw image vs YOLO vs segmentation vs costmap

### Chapter Markers
Add YouTube chapters (timestamps) matching the outline sections above. This dramatically improves viewer retention.

### Description Template
```
Build a complete autonomous perception pipeline with ROS2 + Unity!

In this video: [specific video description]

Code:
- ROS2: https://github.com/Bluebottle10/autonomous-perception-ros2
- Unity: https://github.com/Bluebottle10/autonomous-perception-unity

Technologies: ROS2 Jazzy, TensorRT, YOLO v11, PIDNet, Unity 6, CuPy

Timestamps:
[paste chapter markers here]

#ROS2 #Robotics #AutonomousDriving #TensorRT #Unity #ComputerVision
```
