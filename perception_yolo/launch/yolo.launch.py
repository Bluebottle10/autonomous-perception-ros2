"""
Launch file for YOLOv11 TensorRT detection node.

Usage:
    ros2 launch perception_yolo yolo.launch.py
    ros2 launch perception_yolo yolo.launch.py engine_path:=/path/to/custom.engine
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Locate the default config file
    pkg_dir = get_package_share_directory("perception_yolo")
    default_params = os.path.join(pkg_dir, "config", "yolo_params.yaml")

    return LaunchDescription(
        [
            # Allow overriding the params file from the command line
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the YOLO parameter YAML file",
            ),
            # Launch the YOLO TensorRT node
            Node(
                package="perception_yolo",
                executable="yolo_tensorrt_node",
                name="yolov11_tensorrt_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
