"""
Launch file for PIDNet semantic segmentation node.

Usage:
    ros2 launch perception_segmentation segmentation.launch.py
    ros2 launch perception_segmentation segmentation.launch.py params_file:=/path/to/custom.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("perception_segmentation")
    default_params = os.path.join(pkg_dir, "config", "segmentation_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the segmentation parameter YAML file",
            ),
            Node(
                package="perception_segmentation",
                executable="pidnet_tensorrt_node",
                name="pidnet_tensorrt_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
