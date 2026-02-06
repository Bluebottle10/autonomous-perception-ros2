"""
Full Perception Pipeline Launch File
======================================
Launches the complete autonomous perception stack:

    1. YOLOv11 TensorRT Object Detection
    2. PIDNet Semantic Segmentation
    3. Semantic Costmap Generation
    4. PointCloud Fusion + Visual Overlay
    5. TF Broadcaster
    6. Semantic Driver (optional)

Usage:
    # Full pipeline (all nodes)
    ros2 launch perception_bringup full_pipeline.launch.py

    # Perception only (no driver)
    ros2 launch perception_bringup full_pipeline.launch.py enable_driver:=false

    # Custom config
    ros2 launch perception_bringup full_pipeline.launch.py yolo_params:=/path/to/yolo.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Package directories ──────────────────────────────────────────────
    yolo_dir = get_package_share_directory("perception_yolo")
    seg_dir = get_package_share_directory("perception_segmentation")
    costmap_dir = get_package_share_directory("perception_costmap")
    fusion_dir = get_package_share_directory("perception_fusion")
    nav_dir = get_package_share_directory("perception_navigation")

    # ── Launch arguments ─────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("enable_driver", default_value="true",
                              description="Enable the semantic driver node"),
        DeclareLaunchArgument("enable_lidar", default_value="true",
                              description="Enable the LiDAR listener node"),
    ]

    # ── Include sub-launch files ─────────────────────────────────────────
    includes = [
        # 1. YOLO Detection
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(yolo_dir, "launch", "yolo.launch.py")
            )
        ),
        # 2. Semantic Segmentation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(seg_dir, "launch", "segmentation.launch.py")
            )
        ),
        # 3. Costmap Generation
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(costmap_dir, "launch", "costmap.launch.py")
            )
        ),
        # 4. Fusion (PointCloud + Visual)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(fusion_dir, "launch", "fusion.launch.py")
            )
        ),
    ]

    # ── Conditional nodes ────────────────────────────────────────────────
    nav_params = os.path.join(nav_dir, "config", "navigation_params.yaml")

    conditional_nodes = [
        # TF Broadcaster (always on)
        Node(
            package="perception_navigation",
            executable="tf_broadcaster_node",
            name="tf_broadcaster_node",
            output="screen",
            parameters=[nav_params],
        ),
        # Semantic Driver (optional)
        Node(
            package="perception_navigation",
            executable="semantic_driver_node",
            name="semantic_driver_node",
            output="screen",
            parameters=[nav_params],
            condition=IfCondition(LaunchConfiguration("enable_driver")),
        ),
        # LiDAR Listener (optional)
        Node(
            package="perception_navigation",
            executable="lidar_listener_node",
            name="lidar_listener_node",
            output="screen",
            parameters=[nav_params],
            condition=IfCondition(LaunchConfiguration("enable_lidar")),
        ),
    ]

    return LaunchDescription(args + includes + conditional_nodes)
