"""Spatial visualization functions."""

from typing import Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as scp
import muon as mu


def plot_umap_components_and_rgb(
    colors: np.ndarray,
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    img: np.ndarray,
    spatial_shape: Tuple[int, int] = (50, 50),
    extent: Tuple[int, int, int, int] = (0, 2174, 0, 2174),
    titles: Tuple[str, str, str, str] = ('UMAP 1', 'UMAP 2', 'UMAP 3', 'Combined RGB'),
    figsize: Tuple[int, int] = (20, 5),
    alpha_overlay: float = 0.2,
    alpha_background: float = 0.8
) -> plt.Figure:
    """
    Plot three color components and a combined RGB overlay on a background image.

    Args:
        colors: Color array of shape (n_points, 3), each column is a component.
        x_pixels: x pixel coordinates for each point.
        y_pixels: y pixel coordinates for each point.
        img: Background image (as loaded by mpimg.imread).
        spatial_shape: Shape of the spatial grid (height, width).
        extent: Matplotlib extent for imshow.
        titles: Titles for the subplots.
        figsize: Figure size.
        alpha_overlay: Alpha value for the overlay.
        alpha_background: Alpha value for the background image.
        
    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    n_comp = colors.shape[1]
    grid = [np.zeros(spatial_shape) for _ in range(n_comp)]
    rgb_grid = np.zeros(spatial_shape + (3,))
    
    # Fill grids
    for i in range(len(colors)):
        x, y = x_pixels[i], y_pixels[i]
        if 0 <= x < spatial_shape[0] and 0 <= y < spatial_shape[1]:
            for c in range(n_comp):
                grid[c][x, y] = colors[i, c]
            rgb_grid[x, y] = colors[i]
    
    cmaps = ['Reds', 'Greens', 'Blues']
    for c in range(min(n_comp, 3)):  # Handle cases with fewer than 3 components
        axes[c].imshow(grid[c], extent=extent, cmap=cmaps[c])
        axes[c].set_title(titles[c])
        axes[c].axis('off')
    
    # Combined RGB plot
    axes[3].imshow(img, alpha=alpha_background, extent=extent)
    axes[3].imshow(rgb_grid, extent=extent, alpha=alpha_overlay)
    axes[3].set_title(titles[3])
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_spatial_gene_expression(
    adata: mu.MuData,
    gene_name: str,
    layer: str = 'M_s',
    img_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = 'viridis',
    alpha_overlay: float = 0.7,
    alpha_background: float = 0.8,
    spatial_shape: Tuple[int, int] = (50, 50),
    extent: Tuple[int, int, int, int] = (0, 2174, 0, 2174)
) -> plt.Figure:
    """
    Plot spatial gene expression on tissue image.
    
    Args:
        adata: MuData object containing the data.
        gene_name: Name of the gene to plot.
        layer: Layer containing expression data.
        img_path: Path to background tissue image.
        figsize: Figure size.
        cmap: Colormap for expression values.
        alpha_overlay: Alpha value for expression overlay.
        alpha_background: Alpha value for background image.
        spatial_shape: Shape of the spatial grid.
        extent: Matplotlib extent for imshow.
        
    Returns:
        matplotlib Figure object.
    """
    if gene_name not in adata['rna'].var.index:
        raise ValueError(f"Gene '{gene_name}' not found in data")
    
    # Get gene expression and spatial coordinates
    gene_idx = adata['rna'].var.index.get_loc(gene_name)
    if hasattr(adata['rna'].layers[layer], 'toarray'):
        expression = adata['rna'].layers[layer][:, gene_idx].toarray().flatten()
    else:
        expression = adata['rna'].layers[layer][:, gene_idx]
    
    x_pixels = adata.obs['x_pixel'].values.astype(int)
    y_pixels = adata.obs['y_pixel'].values.astype(int)
    
    # Create spatial expression grid
    expr_grid = np.zeros(spatial_shape)
    for i in range(len(expression)):
        x, y = x_pixels[i], y_pixels[i]
        if 0 <= x < spatial_shape[0] and 0 <= y < spatial_shape[1]:
            expr_grid[x, y] = expression[i]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Expression heatmap
    im1 = axes[0].imshow(expr_grid, extent=extent, cmap=cmap)
    axes[0].set_title(f'{gene_name} Expression')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Overlay on tissue image
    if img_path and img_path.exists():
        img = mpimg.imread(img_path)
        img = np.fliplr(img)  # Flip to match coordinate system
        axes[1].imshow(img, alpha=alpha_background, extent=extent)
    
    im2 = axes[1].imshow(expr_grid, extent=extent, cmap=cmap, alpha=alpha_overlay)
    axes[1].set_title(f'{gene_name} on Tissue')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.tight_layout()
    return fig


def plot_velocity_field(
    adata: mu.MuData,
    velocity_u: np.ndarray,
    velocity_s: np.ndarray,
    subsample: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    arrow_scale: float = 1.0,
    alpha: float = 0.7
) -> plt.Figure:
    """
    Plot RNA velocity field in spatial coordinates.
    
    Args:
        adata: MuData object containing spatial coordinates.
        velocity_u: Unspliced velocity vectors.
        velocity_s: Spliced velocity vectors.
        subsample: Number of cells to subsample for plotting.
        figsize: Figure size.
        arrow_scale: Scale factor for velocity arrows.
        alpha: Alpha value for arrows.
        
    Returns:
        matplotlib Figure object.
    """
    # Subsample cells for visualization
    n_cells = min(subsample, adata.n_obs)
    indices = np.random.choice(adata.n_obs, n_cells, replace=False)
    
    x_pos = adata.obs['x_position'].iloc[indices].values
    y_pos = adata.obs['y_position'].iloc[indices].values
    
    # Compute velocity magnitude and direction
    vel_u_sub = velocity_u[indices]
    vel_s_sub = velocity_s[indices]
    
    # Average velocity across genes for visualization
    vel_u_avg = np.mean(vel_u_sub, axis=1)
    vel_s_avg = np.mean(vel_s_sub, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot velocity field
    ax.quiver(
        x_pos, y_pos, vel_u_avg, vel_s_avg,
        scale=arrow_scale, alpha=alpha, width=0.003
    )
    
    # Plot cell positions
    ax.scatter(x_pos, y_pos, c='black', s=10, alpha=0.5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('RNA Velocity Field')
    
    plt.tight_layout()
    return fig