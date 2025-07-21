"""Visualization utility functions."""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from umap import UMAP
from sklearn.decomposition import PCA
import scipy as scp


def create_umap_embedding(
    data: Union[np.ndarray, torch.Tensor, scp.sparse.spmatrix],
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: Optional[int] = 42
) -> np.ndarray:
    """
    Create UMAP embedding from high-dimensional data.
    
    Args:
        data: Input data of shape (n_samples, n_features).
        n_components: Number of UMAP components.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance parameter for UMAP.
        metric: Distance metric for UMAP.
        random_state: Random state for reproducibility.
        
    Returns:
        UMAP embedding of shape (n_samples, n_components).
    """
    # Convert to numpy array if needed
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    elif hasattr(data, 'toarray'):
        data = data.toarray()
    
    # Ensure data is float32 for UMAP
    data = data.astype(np.float32)
    
    # Create UMAP embedding
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    embedding = umap_model.fit_transform(data)
    return embedding


def create_pca_embedding(
    data: Union[np.ndarray, torch.Tensor, scp.sparse.spmatrix],
    n_components: int = 3,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create PCA embedding from high-dimensional data.
    
    Args:
        data: Input data of shape (n_samples, n_features).
        n_components: Number of PCA components.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple of (PCA embedding, explained variance ratio).
    """
    # Convert to numpy array if needed
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    elif hasattr(data, 'toarray'):
        data = data.toarray()
    
    # Ensure data is float64 for PCA
    data = data.astype(np.float64)
    
    # Create PCA embedding
    pca_model = PCA(n_components=n_components, random_state=random_state)
    embedding = pca_model.fit_transform(data)
    
    return embedding, pca_model.explained_variance_ratio_


def normalize_colors(
    embedding: np.ndarray,
    method: str = 'softmax'
) -> np.ndarray:
    """
    Normalize embedding values to create color values.
    
    Args:
        embedding: Embedding array of shape (n_samples, n_components).
        method: Normalization method ('softmax', 'minmax', or 'standard').
        
    Returns:
        Normalized color array suitable for RGB visualization.
    """
    if method == 'softmax':
        # Apply softmax along each dimension
        colors = scp.special.softmax(embedding, axis=1)
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = embedding.min(axis=0, keepdims=True)
        max_vals = embedding.max(axis=0, keepdims=True)
        colors = (embedding - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == 'standard':
        # Standard normalization then sigmoid
        mean_vals = embedding.mean(axis=0, keepdims=True)
        std_vals = embedding.std(axis=0, keepdims=True)
        normalized = (embedding - mean_vals) / (std_vals + 1e-8)
        colors = 1 / (1 + np.exp(-normalized))  # Sigmoid
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Ensure values are in [0, 1] range
    colors = np.clip(colors, 0, 1)
    
    return colors


def create_rgb_colors(
    data: Union[np.ndarray, torch.Tensor, scp.sparse.spmatrix],
    method: str = 'umap',
    normalize: str = 'softmax',
    **kwargs
) -> np.ndarray:
    """
    Create RGB color values from high-dimensional data.
    
    Args:
        data: Input data of shape (n_samples, n_features).
        method: Dimensionality reduction method ('umap' or 'pca').
        normalize: Color normalization method.
        **kwargs: Additional arguments for embedding methods.
        
    Returns:
        RGB color array of shape (n_samples, 3).
    """
    if method == 'umap':
        embedding = create_umap_embedding(data, n_components=3, **kwargs)
    elif method == 'pca':
        embedding, _ = create_pca_embedding(data, n_components=3, **kwargs)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    colors = normalize_colors(embedding, method=normalize)
    
    # Ensure exactly 3 components for RGB
    if colors.shape[1] > 3:
        colors = colors[:, :3]
    elif colors.shape[1] < 3:
        # Pad with zeros if fewer than 3 components
        padding = np.zeros((colors.shape[0], 3 - colors.shape[1]))
        colors = np.concatenate([colors, padding], axis=1)
    
    return colors