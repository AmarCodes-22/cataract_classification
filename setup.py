import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="dojo",
    version="0.0",
    description="MLOps Framework",
    author="Amar",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(open(os.path.join(os.path.dirname(__file__), "requirements.txt")))
    ],
)
