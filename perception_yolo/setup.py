import os
from glob import glob
from setuptools import find_packages, setup

package_name = "perception_yolo"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Install launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        # Install config files
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="YOLOv11 real-time object detection with TensorRT acceleration for ROS2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_tensorrt_node = perception_yolo.yolo_tensorrt_node:main",
        ],
    },
)
