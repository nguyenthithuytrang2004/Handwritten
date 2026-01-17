"""
Setup script for Handwritten Character Recognition System
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="handwritten-character-recognition",
    version="2.0.0",
    description="Handwritten Character Recognition System with CNN and OCR",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Handwritten Recognition Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "jupyter",
            "ipykernel",
        ],
        "gpu": [
            "tensorflow-gpu>=2.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hwr=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
