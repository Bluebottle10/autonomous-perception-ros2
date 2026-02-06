import os
from glob import glob
from setuptools import find_packages, setup

package_name = "perception_costmap"

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
    description="Semantic costmap generation from segmentation and depth for robot navigation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_costmap_node = perception_costmap.semantic_costmap_node:main",
        ],
    },
)
