"""Simplified Tangelo model with population-level dynamics."""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch_geometric.utils
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import torchode as to
import scipy as scp

from .base import MLP, SigmoidFeatureModule, SigmoidFeatureTrainer
from .encoder import GraphVAEncoder
from .batch_velocity import VelocityEncoder, BatchODESolver
from ..data.utils import get_cdf


class SimplifiedVelocityModel(nn.Module):
    """
    Simplified velocity model for batch-level RNA dynamics.
    
    State vector: y = [u, s, c_open] for all cells in batch
    ODE system: 
        du/dt = c_open * (W @ sigmoid(s)) - beta * u + interaction
        ds/dt = beta * u - gamma * s  
        dc_open/dt = 0 (chromatin accessibility remains constant)
    
    Following the pattern from the provided torchode example where all batch 
    cells are simulated together since they share the same ODE parameters.
    
    Args:
        num_genes: Number of genes.
        sigmoid_function: Pre-trained sigmoid module.
        W: Gene interaction weight matrix (shared across batch).
        interaction: Batch-specific interaction terms.
        beta: Batch-specific beta parameters.
        gamma: Batch-specific gamma parameters.
    """
    
    def __init__(
        self,
        num_genes: int,
        sigmoid_function: SigmoidFeatureModule,
        W: torch.Tensor,
        interaction: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.sigmoid = sigmoid_function
        self.W = W  # Shape: (num_genes, num_genes) - shared across batch
        self.interaction = interaction  # Shape: (batch_size, num_genes)
        self.beta = beta  # Shape: (batch_size, num_genes)
        self.gamma = gamma  # Shape: (batch_size, num_genes)

    def forward(self, t: Union[float, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Computes the velocity vector dy/dt for batch dynamics.

        Args:
            t: Current time (often unused in autonomous systems).
            y: State tensor of shape (batch_size, 3*num_genes) 
               representing [u, s, c_open] for all cells in batch.

        Returns:
            The velocity vector dy/dt with same shape as y.
        """
        # Reshape state: y is (batch_size, 3*num_genes)
        u = y[:, :self.num_genes]  # (batch_size, num_genes)
        s = y[:, self.num_genes:2*self.num_genes]  # (batch_size, num_genes)
        c_open = y[:, 2*self.num_genes:]  # (batch_size, num_genes)
        
        # Apply sigmoid to spliced counts
        sigma = self.sigmoid(s)  # (batch_size, num_genes)
        
        # Gene interaction: sigma @ W.T -> (batch_size, num_genes)
        W_sigma = torch.matmul(sigma, self.W.T)  # (batch_size, num_genes)
        
        # Velocity equations (c_open modulates W interaction, no separate c parameter)
        du_dt = c_open * W_sigma - self.beta * u + self.interaction
        ds_dt = self.beta * u - self.gamma * s
        dc_open_dt = torch.zeros_like(c_open)  # Chromatin accessibility is constant
    
        return torch.cat([du_dt, ds_dt, dc_open_dt], dim=1)


class SimplifiedTangeloModel(nn.Module):
    """
    Simplified Tangelo model with population-level dynamics and single W matrix.
    
    Key simplifications:
    1. No ATAC peaks in input (only RNA + chromatin accessibility + spatial)
    2. Single shared W matrix instead of mixture components  
    3. Population-level ODE simulation (all cells together)
    4. Chromatin accessibility (c_open) as part of state with zero velocity
    5. Zero initial conditions (can be made trainable later)
    
    Args:
        gene_dim: Number of genes.
        spatial_dim: Number of spatial dimensions.
        activation_fn: Activation function name.
        batch_norm: Whether to use batch normalization.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
        hidden_dim_expression: Hidden dimension for expression branch.
        hidden_dim_spatial: Hidden dimension for spatial branch.
        hidden_dim_decoder: Hidden dimension for decoders.
        latent_dim: Latent space dimension.
        gnn_layers: Number of GNN layers.
        mlp_layers: Number of MLP layers.
        n_neighbors: Number of neighbors for graph construction.
        tangent_loss_kwargs: Arguments for tangent loss computation.
    """
    
    def __init__(
        self,
        gene_dim: int,
        spatial_dim: int,
        activation_fn: str,
        batch_norm: bool,
        dropout: float,
        residual: bool,
        hidden_dim_expression: int,
        hidden_dim_spatial: int,
        hidden_dim_decoder: int,
        latent_dim: int,
        gnn_layers: int,
        mlp_layers: int,
        n_neighbors: int,
        tangent_loss_kwargs: Dict[str, float]
    ) -> None:
        super().__init__()
        
        self.gene_dim = gene_dim
        self.spatial_dim = spatial_dim
        # Simplified input: [u, s, c_open, spatial] (no ATAC peaks)
        self.input_dim = 2 * gene_dim + spatial_dim
        self.latent_dim = latent_dim
        self.n_neighbors = n_neighbors
        self.tangent_loss_kwargs = tangent_loss_kwargs

        # Graph VAE encoder (same as before but with simplified input)
        self.graph_va_encoder = GraphVAEncoder(
            self.input_dim, hidden_dim_spatial, hidden_dim_expression,
            latent_dim, gnn_layers, activation_fn, batch_norm, dropout, residual
        )

        # Parameter decoders (per-cell parameters)
        self.beta_decoder = MLP(
            latent_dim, hidden_dim_decoder, gene_dim, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
        self.gamma_decoder = MLP(
            latent_dim, hidden_dim_decoder, gene_dim, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
        self.time_encoder = MLP(
            latent_dim, hidden_dim_decoder, 1, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
        self.decoder_interaction = MLP(
            latent_dim, hidden_dim_decoder, gene_dim, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )

        # Single shared W matrix (not a mixture anymore)
        self.W = nn.Parameter(torch.randn(gene_dim, gene_dim) * 0.01)
        
        # Pre-trained sigmoid function
        self.sigmoid_function = SigmoidFeatureModule(gene_dim)
        
        # Batch velocity encoder following VELOVI pattern
        self.velocity_encoder = VelocityEncoder(
            gene_dim, self.sigmoid_function, self.W
        )
        
        # ODE solver for batch processing
        self.ode_solver = BatchODESolver(
            self.velocity_encoder, gene_dim, dt0=torch.ones(1)
        )
        
        # Scale parameters for likelihood
        self.scale_unconstrained = nn.Parameter(-1.0 * torch.ones(2 * gene_dim))
        
        # Base decoder for tangent loss
        self.base_decoder = MLP(
            latent_dim, hidden_dim_decoder, n_neighbors, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
        
        # Buffers for distance matrix and ODE solver
        self.register_buffer('dist_matrix', torch.empty(0))
        self.register_buffer("dt0", torch.ones([1]))

    def forward(
        self,
        x: torch.Tensor,
        c_open: torch.Tensor,
        space_edge_index: torch.Tensor,
        expression_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the simplified Tangelo model.
        
        Args:
            x: Input features [u, s, spatial_coords].
            c_open: Open chromatin features.
            space_edge_index: Spatial graph edge indices.
            expression_edge_index: Expression graph edge indices.
            
        Returns:
            Tuple of (pred_u, pred_s, qz_mean, qz_log_var).
        """
        # Encode latent representations
        qz_mean, qz_log_var, z, z_spatial, z_expression = self.graph_va_encoder(
            x, space_edge_index, expression_edge_index
        )

        # Decode per-cell parameters
        beta = F.softplus(self.beta_decoder(z))
        gamma = F.softplus(self.gamma_decoder(z))
        interaction = self.decoder_interaction(z_spatial)
        t = F.softplus(self.time_encoder(z))

        # Zero initial conditions for now
        batch_size = x.shape[0]
        x0 = torch.zeros(batch_size, 3 * self.gene_dim, device=x.device)
        # Set c_open in initial conditions
        x0[:, 2*self.gene_dim:] = c_open

        # Simulate batch-level ODE
        pred_u, pred_s = self.simulate(t, x0, interaction, beta, gamma, batch_size)

        return pred_u, pred_s, qz_mean, qz_log_var
    
    def loss(
        self,
        u_true: torch.Tensor,
        s_true: torch.Tensor,
        pred_u: torch.Tensor,
        pred_s: torch.Tensor,
        qz_mean: torch.Tensor,
        qz_log_var: torch.Tensor,
        expression_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the total loss for the simplified model."""
        # Reconstruction loss
        scale = F.softplus(self.scale_unconstrained)
        dist_u = Normal(pred_u, scale[:self.gene_dim])
        dist_s = Normal(pred_s, scale[self.gene_dim:])
        recon_loss_u = -dist_u.log_prob(u_true).sum(dim=1)
        recon_loss_s = -dist_s.log_prob(s_true).sum(dim=1)
        reconstruction_loss = recon_loss_u + recon_loss_s

        # KL divergence
        prior = Normal(0, 1)
        posterior = Normal(qz_mean, torch.exp(0.5 * qz_log_var))
        kl_divergence_z = kl(posterior, prior).sum(dim=1)

        # Velocity tangent loss (simplified)
        base = self.base_decoder(qz_mean)
        velocity_tangent_loss = self.velocity_tangent_loss(base, expression_edge_index)

        return (reconstruction_loss + kl_divergence_z + velocity_tangent_loss).mean()
    
    def velocity_tangent_loss(
        self,
        base: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Simplified tangent loss computation."""
        try:
            row, col = edge_index
            dist_matrix_device = self.dist_matrix.to(row.device)
            
            P_not_norm = dist_matrix_device[row, col] 
            P_norm = P_not_norm / P_not_norm.sum(dim=0, keepdim=True).clamp(min=1e-8)
            
            projection_loss = F.cosine_similarity(base, P_norm, dim=1)
            reg_loss = torch.norm(base, p=2, dim=1).pow(2)
            
            lambda_reg = self.tangent_loss_kwargs.get('lambda_reg', 0.1)
            return -projection_loss + lambda_reg * reg_loss
            
        except (IndexError, RuntimeError):
            return torch.tensor(0.0, device=base.device)
        
    def simulate(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        interaction: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch-level ODE simulation using VELOVI pattern with shared velocity encoder.
        
        KEY CHANGE: Following VELOVI pattern, beta and gamma must be shared across
        the entire batch for the ODE solver to work correctly. Only interaction
        can be cell-specific.
        
        Args:
            t: Time points for each cell (batch_size,).
            x0: Initial conditions [u0, s0, c_open] (batch_size, 3*gene_dim).
            interaction: Cell-specific interaction terms (batch_size, gene_dim).
            beta: Shared beta parameters (gene_dim,) - SAME for all cells.
            gamma: Shared gamma parameters (gene_dim,) - SAME for all cells.
            batch_size: Number of cells.
            
        Returns:
            Tuple of (pred_u, pred_s).
        """
        # Following VELOVI pattern: use shared velocity encoder with batch solver
        # beta and gamma are now shared across the batch (not per-cell)
        shared_beta = beta.mean(dim=0) if beta.dim() > 1 else beta
        shared_gamma = gamma.mean(dim=0) if gamma.dim() > 1 else gamma
        shared_interaction = interaction.mean(dim=0) if interaction.dim() > 1 else interaction
        
        return self.ode_solver.solve_batch(
            t, x0, shared_interaction, shared_beta, shared_gamma
        )
    
    def pretrain_sigmoid(
        self,
        s: torch.Tensor,
        n_epochs: int = 100,
        learning_rate: float = 1.0
    ) -> None:
        """Pre-trains the sigmoid function on the data CDF."""
        x_cdf, y_cdf = get_cdf(s)
        
        model_to_train = SigmoidFeatureModule(self.gene_dim).to(s.device)
        trainer = SigmoidFeatureTrainer(model_to_train, learning_rate=learning_rate)
        trainer.fit(x_cdf.T, y_cdf.T, epochs=n_epochs, verbose=True)
        
        self.sigmoid_function.set_parameters(model_to_train.get_parameters())
        for param in self.sigmoid_function.parameters():
            param.requires_grad = False
        print("Sigmoid pre-training complete.")