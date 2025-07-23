"""Preprocessing functions for Tangelo."""

# Import preprocessing utilities
from ..data import (
    create_peak_by_cell_matrix,
    setup_multimodal_data,
)
from ..plot.utils import (
    create_umap_embedding,
    create_pca_embedding,
    normalize_colors,
)

__all__ = [
    "create_peak_by_cell_matrix",
    "setup_multimodal_data",
    "create_umap_embedding",
    "create_pca_embedding", 
    "normalize_colors",
]