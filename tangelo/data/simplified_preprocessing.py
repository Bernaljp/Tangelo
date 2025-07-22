"""Simplified data preprocessing for the new Tangelo architecture."""

from typing import Tuple, Optional
import numpy as np
import torch
import scipy as scp
from sklearn.neighbors import kneighbors_graph
import muon as mu


def setup_simplified_data(
    adata: mu.MuData,
    n_neighbors_spatial: int = 8,
    n_neighbors_expression: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares simplified data for the new Tangelo architecture.
    
    Key changes:
    1. No ATAC peaks in input matrix
    2. Input is now [U, S, c_open, spatial_coordinates]
    3. Simpler data structure for population-level simulation
    
    Args:
        adata: MuData object containing multi-modal data.
        n_neighbors_spatial: Number of spatial neighbors for graph.
        n_neighbors_expression: Number of expression neighbors for graph.
        
    Returns:
        Tuple of (input_matrix, c_open_matrix, spatial_edge_index, expression_edge_index).
    """
    print("Setting up simplified data (no ATAC peaks)...")
    
    # Extract RNA data
    U = adata['rna'].layers['M_u']  # Unspliced
    S = adata['rna'].layers['M_s']  # Spliced  
    C_open = adata['rna'].layers['open_chromatin']  # Chromatin accessibility
    
    # Extract spatial coordinates
    spatial_coords = scp.sparse.csr_matrix(
        adata.obs[['x_position', 'y_position']].values
    )
    
    # Create simplified input matrix: [U, S, spatial_coordinates]
    # Note: c_open is returned separately as it has special treatment
    X_simplified = scp.sparse.hstack([U, S, spatial_coords]).tocsr()
    
    print(f"Simplified input shape: {X_simplified.shape}")
    print(f"Components: U({U.shape[1]}), S({S.shape[1]}), spatial({spatial_coords.shape[1]})")
    
    # Create spatial graph (same as before)
    spatial_coords_dense = adata.obs[['x_position', 'y_position']].values
    spatial_graph = kneighbors_graph(
        spatial_coords_dense, n_neighbors_spatial, 
        mode='connectivity', include_self=False
    )
    spatial_edge_coo = spatial_graph.tocoo()
    spatial_edge_index = torch.tensor(
        np.vstack((spatial_edge_coo.row, spatial_edge_coo.col)), 
        dtype=torch.long
    )
    
    # Create expression graph using only RNA data (no ATAC)
    expr_data = S.toarray() if hasattr(S, 'toarray') else S
    expr_graph = kneighbors_graph(
        expr_data, n_neighbors_expression,
        mode='connectivity', include_self=False
    )
    expr_edge_coo = expr_graph.tocoo()
    expression_edge_index = torch.tensor(
        np.vstack((expr_edge_coo.row, expr_edge_coo.col)),
        dtype=torch.long
    )
    
    print(f"Spatial graph: {spatial_edge_index.shape[1]} edges")
    print(f"Expression graph: {expression_edge_index.shape[1]} edges")
    
    return X_simplified, C_open, spatial_edge_index, expression_edge_index


def create_simplified_batch_data(
    X: scp.sparse.csr_matrix,
    C_open: scp.sparse.csr_matrix,
    indices: np.ndarray,
    gene_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates batch data for simplified model training.
    
    Args:
        X: Simplified input matrix [U, S, spatial].
        C_open: Chromatin accessibility matrix.
        indices: Batch indices.
        gene_dim: Number of genes.
        device: Target device.
        
    Returns:
        Tuple of (batch_input, c_open_batch, u_batch, s_batch).
    """
    # Extract batch data
    batch_x = torch.tensor(X[indices].toarray().astype(np.float32), device=device)
    c_open_batch = torch.tensor(C_open[indices].toarray().astype(np.float32), device=device)
    
    # Split into components
    u_batch = batch_x[:, :gene_dim]
    s_batch = batch_x[:, gene_dim:2*gene_dim]
    spatial_batch = batch_x[:, 2*gene_dim:]
    
    # Create input for model: [U, S, spatial] (c_open handled separately)
    batch_input = torch.cat([u_batch, s_batch, spatial_batch], dim=1)
    
    return batch_input, c_open_batch, u_batch, s_batch


def validate_simplified_data(
    X: scp.sparse.csr_matrix,
    C_open: scp.sparse.csr_matrix,
    gene_dim: int
) -> None:
    """
    Validates the simplified data format.
    
    Args:
        X: Input matrix.
        C_open: Chromatin accessibility matrix.
        gene_dim: Expected number of genes.
    """
    print("\n=== Data Validation ===")
    
    # Check dimensions
    expected_input_dim = 2 * gene_dim + 2  # U + S + spatial(x,y)
    assert X.shape[1] == expected_input_dim, f"Expected {expected_input_dim} input features, got {X.shape[1]}"
    
    assert C_open.shape[1] == gene_dim, f"Expected {gene_dim} genes in C_open, got {C_open.shape[1]}"
    
    assert X.shape[0] == C_open.shape[0], f"Mismatched number of cells: X({X.shape[0]}) vs C_open({C_open.shape[0]})"
    
    # Check data types and ranges
    print(f"✅ Input matrix shape: {X.shape}")
    print(f"✅ C_open matrix shape: {C_open.shape}")
    print(f"✅ Gene dimension: {gene_dim}")
    
    # Check for non-negative values (RNA counts should be non-negative)
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
        
    u_data = X_dense[:, :gene_dim]
    s_data = X_dense[:, gene_dim:2*gene_dim]
    
    assert np.all(u_data >= 0), "Unspliced counts contain negative values"
    assert np.all(s_data >= 0), "Spliced counts contain negative values"
    
    print(f"✅ U range: [{u_data.min():.3f}, {u_data.max():.3f}]")
    print(f"✅ S range: [{s_data.min():.3f}, {s_data.max():.3f}]")
    
    # Check sparsity
    u_sparsity = np.mean(u_data == 0)
    s_sparsity = np.mean(s_data == 0)
    
    print(f"✅ U sparsity: {u_sparsity:.1%}")
    print(f"✅ S sparsity: {s_sparsity:.1%}")
    
    print("=== Validation Complete ===\n")


def compare_data_sizes(adata_original: mu.MuData, X_simplified: scp.sparse.csr_matrix) -> None:
    """
    Compares data sizes between original and simplified versions.
    
    Args:
        adata_original: Original MuData with ATAC peaks.
        X_simplified: Simplified input matrix without ATAC peaks.
    """
    print("\n=== Data Size Comparison ===")
    
    # Original data size calculation
    original_rna_features = adata_original['rna'].n_vars
    original_atac_features = adata_original['atac'].n_vars if 'atac' in adata_original.mod else 0
    original_spatial_features = 2
    original_total = 2 * original_rna_features + original_rna_features + original_atac_features + original_spatial_features
    
    # Simplified data size
    simplified_total = X_simplified.shape[1] + original_rna_features  # X + C_open
    
    print(f"Original model input size: {original_total:,}")
    print(f"  - RNA (U+S): {2 * original_rna_features:,}")
    print(f"  - RNA (C_open): {original_rna_features:,}")
    print(f"  - ATAC peaks: {original_atac_features:,}")
    print(f"  - Spatial: {original_spatial_features}")
    
    print(f"\nSimplified model input size: {simplified_total:,}")
    print(f"  - RNA (U+S): {2 * original_rna_features:,}")
    print(f"  - RNA (C_open): {original_rna_features:,} (separate)")
    print(f"  - ATAC peaks: 0")
    print(f"  - Spatial: {original_spatial_features}")
    
    reduction = (original_total - simplified_total) / original_total
    print(f"\n✅ Size reduction: {reduction:.1%}")
    print(f"✅ Simplified model is {simplified_total/original_total:.1%} the size of original")
    
    print("=== Comparison Complete ===\n")