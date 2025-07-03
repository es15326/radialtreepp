from setuptools import setup, find_packages
import os

setup(
    name="radiatreepp",
    version="0.1",
    author="Elham Soltani Kazemi",
    description="Radial dendrogram visualization with feature annotations",
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
