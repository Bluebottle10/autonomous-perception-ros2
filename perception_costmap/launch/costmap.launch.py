"""
Launch file for the Semantic Costmap Generation node.

Usage:
    ros2 launch perception_costmap costmap.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("perception_costmap")
    default_params = os.path.join(pkg_dir, "config", "costmap_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the costmap parameter YAML file",
            ),
            Node(
                package="perception_costmap",
                executable="semantic_costmap_node",
                name="semantic_costmap_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
