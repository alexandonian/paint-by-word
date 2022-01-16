#!/usr/bin/env python3

# If using pip, this enables, for example, `pip install -v -e .`

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paintbyword",
    version="0.0.1",
    author="Alex Andonian, David Bau",
    author_email="andonian@csail.mit.edu",
    description="Paint By Word",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexandonian/paintbyword",
    packages=['paintbyword'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ),
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.1",
        "torchvision>=0.8.2",
        "pyyaml>=3.12",
        "numpy>=1.14.5",
        "Pillow>=4.1.0",
        "scipy>=1.1.0",
        "scikit-image>=0.14.0",
        "tqdm>=4.23.4",
        "cma>=3.0.3"
    ],
)
