"""Setup script for the Tangelo package."""

from setuptools import setup, find_packages
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="tangelo",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning for Multi-Modal Single-Cell Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tangelo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=1.12.0",
        "torch-geometric>=2.0.0",
        "torchode>=0.2.0",
        
        # Scientific computing
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        
        # Single-cell analysis
        "scanpy>=1.8.0",
        "muon>=0.1.0",
        "dynamo-release>=1.2.0",
        
        # Bioinformatics
        "pysam>=0.19.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "umap-learn>=0.5.0",
        
        # Utilities
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "tangelo=tangelo.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)