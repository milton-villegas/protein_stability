"""
Setup script for Protein Stability DoE Analysis Tool
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="protein-stability-doe",
    version="0.2.0",
    description="Design of Experiments (DoE) analysis tool for protein stability studies",
    author="Milton F. Villegas",
    author_email="miltonfvillegas@gmail.com",
    python_requires=">=3.10,<4.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0,<10.0.0",
            "pytest-cov>=4.0.0,<8.0.0",
            "pytest-mock>=3.10.0,<4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
