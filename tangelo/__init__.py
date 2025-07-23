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

# Modular imports following scanpy-style organization
from . import data
from . import plot  
from . import pp
from . import tools

# Convenient top-level access to key functions
from .data import setup_multimodal_data
from .training import TangeloTrainer
from .config import TangeloConfig, load_config, save_config

# Keep some compatibility imports for existing code
from .data import (
    create_peak_by_cell_matrix,
    setup_multimodal_data,
)
from .tools import (
    create_graph_data,
    get_cdf,
)
from .plot import (
    plot_umap_components_and_rgb,
    plot_spatial_gene_expression,
    create_umap_embedding,
    create_rgb_colors,
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
    # Modular access
    "data",
    "plot",
    "pp", 
    "tools",
    # Top-level functions (for compatibility)
    "setup_multimodal_data",
    "TangeloTrainer",
    "TangeloConfig",
    "load_config",
    "save_config",
    # Legacy compatibility - data functions
    "create_peak_by_cell_matrix",
    "setup_multimodal_data",
    # Legacy compatibility - tools functions  
    "create_graph_data", 
    "get_cdf",
    # Legacy compatibility - plot functions
    "plot_umap_components_and_rgb",
    "plot_spatial_gene_expression",
    "create_umap_embedding",
    "create_rgb_colors",
]