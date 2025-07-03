'''from setuptools import setup, find_packages
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
)'''


from setuptools import setup, find_packages

setup(
    name='radiatreepp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
    ],
    include_package_data=True,
    author='Elham Soltani Kazemi',
    description='Radial dendrogram visualization for hierarchical clustering with feature importance overlays',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/es15326/radialtreepp.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)

