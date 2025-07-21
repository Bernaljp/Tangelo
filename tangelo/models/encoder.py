"""Graph Variational Autoencoder encoder implementation."""

from typing import Tuple
import torch
import torch.nn as nn
from .base import mySAGEConv


class GraphVAEncoder(nn.Module):
    """
    Graph Variational Autoencoder (VAE) Encoder.
    Uses two separate GraphSAGE branches for spatial and expression data.

    Args:
        input_dim: Input feature dimension.
        hidden_dim_spatial: Hidden dimension for the spatial graph branch.
        hidden_dim_expression: Hidden dimension for the expression graph branch.
        latent_dim: Dimension of the latent space.
        gnn_layers: Number of GNN layers.
        activation_fn: Activation function name.
        batch_norm: Whether to use batch normalization.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim_spatial: int,
        hidden_dim_expression: int,
        latent_dim: int,
        gnn_layers: int,
        activation_fn: str,
        batch_norm: bool,
        dropout: float,
        residual: bool
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        
        output_graph_dim = 2 * latent_dim
        
        self.spatial_graph = mySAGEConv(
            input_dim, hidden_dim_spatial, output_graph_dim, 
            gnn_layers, activation_fn, batch_norm, dropout, residual
        )
        self.gene_graph = mySAGEConv(
            input_dim, hidden_dim_expression, output_graph_dim, 
            gnn_layers, activation_fn, batch_norm, dropout, residual
        )
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Performs the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        expression_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.
        
        Args:
            x: Input features.
            spatial_edge_index: Spatial graph edge indices.
            expression_edge_index: Expression graph edge indices.
            
        Returns:
            Tuple of (z_mean, z_log_var, latent_z, latent_z_spatial, latent_z_expression).
        """
        h1 = self.spatial_graph(x, spatial_edge_index)
        h2 = self.gene_graph(x, expression_edge_index)
        
        h1_mean, h1_log_var = h1.split(self.latent_dim, dim=1)
        h2_mean, h2_log_var = h2.split(self.latent_dim, dim=1)
        
        z_mean = h1_mean + h2_mean
        z_log_var = h1_log_var + h2_log_var
        
        latent_z = self.reparameterize(z_mean, z_log_var)
        latent_z_spatial = self.reparameterize(h1_mean, h1_log_var)
        latent_z_expression = self.reparameterize(h2_mean, h2_log_var)
        
        return z_mean, z_log_var, latent_z, latent_z_spatial, latent_z_expression