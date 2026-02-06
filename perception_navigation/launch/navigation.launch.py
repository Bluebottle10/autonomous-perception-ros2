"""
Launch file for navigation nodes (driver, TF broadcaster, LiDAR listener).

Usage:
    ros2 launch perception_navigation navigation.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("perception_navigation")
    default_params = os.path.join(pkg_dir, "config", "navigation_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Path to the navigation parameter YAML file",
            ),
            Node(
                package="perception_navigation",
                executable="semantic_driver_node",
                name="semantic_driver_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
            Node(
                package="perception_navigation",
                executable="tf_broadcaster_node",
                name="tf_broadcaster_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
            Node(
                package="perception_navigation",
                executable="lidar_listener_node",
                name="lidar_listener_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
