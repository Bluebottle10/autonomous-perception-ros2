import os
from glob import glob
from setuptools import find_packages, setup

package_name = "perception_fusion"

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
    description="Sensor fusion nodes: semantic point cloud generation and visual overlay.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pointcloud_fusion_node = perception_fusion.pointcloud_fusion_node:main",
            "visual_fusion_node = perception_fusion.visual_fusion_node:main",
        ],
    },
)
