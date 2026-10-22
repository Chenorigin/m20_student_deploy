from setuptools import setup
from glob import glob

setup(name="m20_heightmap", version="0.1.0", packages=["m20_heightmap"],
      data_files=[("share/ament_index/resource_index/packages", ["resource/m20_heightmap"]),
                  ("share/m20_heightmap", ["package.xml"]),
                  ("share/m20_heightmap/config", glob("config/*.json"))],
      install_requires=["setuptools", "numpy>=1.17"], zip_safe=True,
      maintainer="M20 deployment", maintainer_email="maintainer@example.com",
      description="CPU point-cloud terrain observations for M20 student v0/v1", license="Apache-2.0",
      entry_points={"console_scripts": ["gos_mapper=m20_heightmap.ros_nodes:gos_main",
                                        "aos_adapter=m20_heightmap.ros_nodes:aos_main",
                                        "gos_native_inputs=m20_heightmap.native_inputs:main",
                                        "base_pose_adapter=m20_heightmap.base_pose:main"]})
