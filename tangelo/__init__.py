"""
Tangelo: Deep Learning for Multi-Modal Single-Cell Analysis

A PyTorch package for integrating spatial transcriptomics, RNA velocity, and ATAC-seq data
using graph neural networks and ordinary differential equation modeling.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core model components
from .models import (
    TangeloModel,
    GraphVAEncoder,
    MLP,
    mySAGEConv,
    SigmoidFeatureModule,
    SigmoidFeatureTrainer,
    VelocityModel,
)

# Data processing utilities
from .data import (
    create_peak_by_cell_matrix,
    setup_multimodal_data,
    create_graph_data,
    get_cdf,
)

# Training utilities
from .training import TangeloTrainer

# Configuration management
from .config import TangeloConfig, load_config, save_config

# Visualization utilities
from .visualization import (
    plot_umap_components_and_rgb,
    plot_spatial_gene_expression,
    create_umap_embedding,
)

__all__ = [
    # Core models
    "TangeloModel",
    "GraphVAEncoder", 
    "MLP",
    "mySAGEConv",
    "SigmoidFeatureModule",
    "SigmoidFeatureTrainer",
    "VelocityModel",
    # Data processing
    "create_peak_by_cell_matrix",
    "setup_multimodal_data",
    "create_graph_data", 
    "get_cdf",
    # Training
    "TangeloTrainer",
    # Configuration
    "TangeloConfig",
    "load_config",
    "save_config",
    # Visualization
    "plot_umap_components_and_rgb",
    "plot_spatial_gene_expression",
    "create_umap_embedding",
]