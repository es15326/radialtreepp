from setuptools import setup, find_packages
import os

setup(
    name="radiatreepp",
    version="0.1",
    author="Your Name",
    description="Radial dendrogram visualization with feature annotations",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
