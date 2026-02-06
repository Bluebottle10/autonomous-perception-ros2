"""
Launch file for fusion nodes (pointcloud + visual overlay).

Usage:
    ros2 launch perception_fusion fusion.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("perception_fusion")
    default_params = os.path.join(pkg_dir, "config", "fusion_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the fusion parameter YAML file",
            ),
            # PointCloud Fusion Node
            Node(
                package="perception_fusion",
                executable="pointcloud_fusion_node",
                name="pointcloud_fusion_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
            # Visual Overlay Node
            Node(
                package="perception_fusion",
                executable="visual_fusion_node",
                name="visual_fusion_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
