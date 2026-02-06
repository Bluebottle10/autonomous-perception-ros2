"""
Perception-Only Launch File
=============================
Launches only the perception nodes (no navigation/driver).
Useful for testing detection and segmentation independently.

Nodes launched:
    1. YOLOv11 TensorRT Object Detection
    2. PIDNet Semantic Segmentation
    3. Visual Fusion (overlay)

Usage:
    ros2 launch perception_bringup perception_only.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    yolo_dir = get_package_share_directory("perception_yolo")
    seg_dir = get_package_share_directory("perception_segmentation")
    fusion_dir = get_package_share_directory("perception_fusion")

    return LaunchDescription(
        [
            # YOLO Detection
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(yolo_dir, "launch", "yolo.launch.py")
                )
            ),
            # Semantic Segmentation
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(seg_dir, "launch", "segmentation.launch.py")
                )
            ),
            # Visual overlay only (from fusion package)
            Node(
                package="perception_fusion",
                executable="visual_fusion_node",
                name="visual_fusion_node",
                output="screen",
                parameters=[
                    os.path.join(fusion_dir, "config", "fusion_params.yaml")
                ],
            ),
        ]
    )
