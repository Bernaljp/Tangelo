"""Plotting and visualization functions for Tangelo."""

# Import all plotting functions directly
from .spatial import (
    plot_umap_components_and_rgb,
    plot_spatial_gene_expression,
    plot_velocity_field,
)
from .utils import (
    create_umap_embedding,
    create_pca_embedding,
    normalize_colors,
    create_rgb_colors,
)

__all__ = [
    # Spatial plotting functions
    "plot_umap_components_and_rgb",
    "plot_spatial_gene_expression",
    "plot_velocity_field",
    # Utility functions for embeddings and colors
    "create_umap_embedding",
    "create_pca_embedding",
    "normalize_colors",
    "create_rgb_colors",
]