import os
from glob import glob
from setuptools import find_packages, setup

package_name = "perception_navigation"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Navigation nodes: semantic driver, TF broadcaster, and LiDAR listener.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_driver_node = perception_navigation.semantic_driver_node:main",
            "tf_broadcaster_node = perception_navigation.tf_broadcaster_node:main",
            "lidar_listener_node = perception_navigation.lidar_listener_node:main",
        ],
    },
)
