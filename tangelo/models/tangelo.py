"""Main Tangelo model implementation."""

from typing import Tuple, Dict, Any, Optional
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

from .base import MLP, SigmoidFeatureModule, SigmoidFeatureTrainer, VelocityModel
from .encoder import GraphVAEncoder
from ..data.utils import get_cdf


class TangeloModel(nn.Module):
    """
    The main Tangelo model, integrating the VAE, ODE solver, and loss functions.
    
    Args:
        gene_dim: Number of genes.
        atac_dim: Number of ATAC features.
        spatial_dim: Number of spatial dimensions.
        activation_fn: Activation function name.
        batch_norm: Whether to use batch normalization.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
        hidden_dim_expression: Hidden dimension for expression branch.
        hidden_dim_spatial: Hidden dimension for spatial branch.
        hidden_dim_decoder: Hidden dimension for decoders.
        latent_dim: Latent space dimension.
        n_components: Number of mixture components.
        gnn_layers: Number of GNN layers.
        mlp_layers: Number of MLP layers.
        n_neighbors: Number of neighbors for graph construction.
        tangent_loss_kwargs: Arguments for tangent loss computation.
    """
    
    def __init__(
        self,
        gene_dim: int,
        atac_dim: int,
        spatial_dim: int,
        activation_fn: str,
        batch_norm: bool,
        dropout: float,
        residual: bool,
        hidden_dim_expression: int,
        hidden_dim_spatial: int,
        hidden_dim_decoder: int,
        latent_dim: int,
        n_components: int,
        gnn_layers: int,
        mlp_layers: int,
        n_neighbors: int,
        tangent_loss_kwargs: Dict[str, float]
    ) -> None:
        super().__init__()
        
        self.gene_dim = gene_dim
        self.spatial_dim = spatial_dim
        self.atac_dim = atac_dim
        self.input_dim = 2 * gene_dim + atac_dim + spatial_dim
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.tangent_loss_kwargs = tangent_loss_kwargs

        self.graph_va_encoder = GraphVAEncoder(
            self.input_dim, hidden_dim_spatial, hidden_dim_expression,
            latent_dim, gnn_layers, activation_fn, batch_norm, dropout, residual
        )

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
        self.decoder_components = MLP(
            latent_dim, hidden_dim_decoder, n_components, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
        self.decoder_interaction = MLP(
            latent_dim, hidden_dim_decoder, gene_dim, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
        self.c_decoder = MLP(
            latent_dim, hidden_dim_decoder, gene_dim, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )

        self.x0_decoders = nn.ModuleList([
            MLP(latent_dim, hidden_dim_decoder, 2 * gene_dim, mlp_layers, 
                activation_fn, batch_norm, dropout, residual)
            for _ in range(n_components)
        ])

        self.Ws = nn.ModuleList([
            nn.Linear(gene_dim, gene_dim, bias=False) for _ in range(n_components)
        ])
        self.sigmoid_function = SigmoidFeatureModule(gene_dim)
        
        self.scale_unconstrained = nn.Parameter(-1.0 * torch.ones(2 * gene_dim))
        self.base_decoder = MLP(
            latent_dim, hidden_dim_decoder, n_neighbors, mlp_layers, 
            activation_fn, batch_norm, dropout, residual
        )
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
        Main forward pass of the Tangelo model.
        
        Args:
            x: Input features.
            c_open: Open chromatin features.
            space_edge_index: Spatial graph edge indices.
            expression_edge_index: Expression graph edge indices.
            
        Returns:
            Tuple of (pred_u, pred_s, qz_mean, qz_log_var).
        """
        qz_mean, qz_log_var, z, z_spatial, z_expression = self.graph_va_encoder(
            x, space_edge_index, expression_edge_index
        )

        component_weights = F.softmax(self.decoder_components(z_expression), dim=-1)
        beta = F.softplus(self.beta_decoder(z))
        gamma = F.softplus(self.gamma_decoder(z))
        interaction = self.decoder_interaction(z_spatial)
        t = F.softplus(self.time_encoder(z))
        c = F.softplus(self.c_decoder(z))

        # Calculate the effective weight tensor for the batch
        all_Ws_weights = torch.stack([w.weight for w in self.Ws], dim=0)
        W_effective = torch.einsum('bc,cde->bde', component_weights, all_Ws_weights)

        # Decode initial state x0
        x0_decoded_list = [dec(z) for dec in self.x0_decoders]
        x0_stack = torch.stack(x0_decoded_list, dim=2)
        z_weights = component_weights.unsqueeze(1)
        x0 = torch.bmm(z_weights, x0_stack.transpose(1, 2)).squeeze(1)

        # Simulate the ODE
        pred_u, pred_s = self.simulate(t, x0, W_effective, interaction, beta, gamma, c)

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
        """
        Calculates the total loss.
        
        Args:
            u_true: True unspliced counts.
            s_true: True spliced counts.
            pred_u: Predicted unspliced counts.
            pred_s: Predicted spliced counts.
            qz_mean: Latent mean.
            qz_log_var: Latent log variance.
            expression_edge_index: Expression graph edge indices.
            
        Returns:
            Total loss tensor.
        """
        scale = F.softplus(self.scale_unconstrained)
        dist_u = Normal(pred_u, scale[:self.gene_dim])
        dist_s = Normal(pred_s, scale[self.gene_dim:])
        recon_loss_u = -dist_u.log_prob(u_true).sum(dim=1)
        recon_loss_s = -dist_s.log_prob(s_true).sum(dim=1)
        reconstruction_loss = recon_loss_u + recon_loss_s

        prior = Normal(0, 1)
        posterior = Normal(qz_mean, torch.exp(0.5 * qz_log_var))
        kl_divergence_z = kl(posterior, prior).sum(dim=1)

        # Create a temporary velocity model for the batch to calculate pred_v
        component_weights = F.softmax(self.decoder_components(qz_mean), dim=-1)
        beta = F.softplus(self.beta_decoder(qz_mean))
        gamma = F.softplus(self.gamma_decoder(qz_mean))
        c = F.softplus(self.c_decoder(qz_mean))
        interaction = self.decoder_interaction(qz_mean)
        
        all_Ws_weights = torch.stack([w.weight for w in self.Ws], dim=0)
        W_effective = torch.einsum('bc,cde->bde', component_weights, all_Ws_weights)
        
        temp_velocity_model = VelocityModel(
            self.gene_dim, self.sigmoid_function, W_effective, 
            interaction, beta, gamma, c
        )
        
        y_for_v = torch.cat([pred_u, pred_s], dim=1)
        pred_v = temp_velocity_model(0, y_for_v)
        
        base = self.base_decoder(qz_mean)
        velocity_tangent_loss = self.velocity_tangent_loss(pred_v, base, expression_edge_index)

        return (reconstruction_loss + kl_divergence_z + velocity_tangent_loss).mean()
    
    def velocity_tangent_loss(
        self,
        pred_v: torch.Tensor,
        base: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the tangent loss for velocity alignment.
        
        Args:
            pred_v: Predicted velocity.
            base: Base vectors.
            edge_index: Graph edge indices.
            
        Returns:
            Tangent loss tensor.
        """
        row, col = edge_index
        dist_matrix_device = self.dist_matrix.to(row.device)
        
        try:
            P_not_norm = dist_matrix_device[row, col] 
            P_norm = P_not_norm / P_not_norm.sum(dim=0, keepdim=True).clamp(min=1e-8) - (1.0 / len(row))
            v_tangent_base = torch.sum(base * dist_matrix_device[row, col], dim=1)
            tangent_loss = torch.norm(pred_v - v_tangent_base.unsqueeze(-1), p=2, dim=1).pow(2)
            projection_loss = F.cosine_similarity(base, P_norm, dim=1)
            reg_loss = torch.norm(base, p=2, dim=1).pow(2)
        except IndexError:
            return torch.tensor(0.0, device=pred_v.device)

        a = self.tangent_loss_kwargs.get('a', 1.0)
        b = self.tangent_loss_kwargs.get('b', 1.0)
        lambda_reg = self.tangent_loss_kwargs.get('lambda_reg', 0.1)

        return a * tangent_loss - b * projection_loss + lambda_reg * reg_loss
        
    def simulate(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        W_effective: torch.Tensor,
        interaction: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves the ODE initial value problem by creating a new VelocityModel
        instance for each sample in the batch.
        
        Args:
            t: Time points.
            x0: Initial conditions.
            W_effective: Effective weight matrices.
            interaction: Interaction terms.
            beta: Beta parameters.
            gamma: Gamma parameters.
            c: C parameters.
            
        Returns:
            Tuple of (pred_u, pred_s).
        """
        t_vec = t.squeeze(-1) if t.dim() > 1 else t.squeeze()
        pred_y_list = []

        for i in range(x0.shape[0]):
            # Create a new, fully parameterized velocity model for this single sample
            velocity_model_i = VelocityModel(
                self.gene_dim, self.sigmoid_function,
                W_effective[i:i+1], interaction[i:i+1], 
                beta[i:i+1], gamma[i:i+1], c[i:i+1]
            )

            # Set up the custom solver
            term = to.ODETerm(velocity_model_i)
            step_method = to.Dopri5(term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)

            y0_sample = x0[i:i+1]
            t_end_sample = t_vec[i].unsqueeze(0)
            t_start_sample = torch.zeros_like(t_end_sample)
            
            # Create the Initial Value Problem for the solver
            ivp = to.InitialValueProblem(
                y0=y0_sample, t_start=t_start_sample, t_end=t_end_sample
            )
            
            # Solve the ODE
            sol = solver.solve(ivp, dt0=self.dt0)
            
            # The solution tensor `sol.y` has shape (time_steps, batch_size, dims).
            # Here batch_size is 1. We want the last time step.
            final_state = sol.ys[:,-1].squeeze(0)
            pred_y_list.append(final_state)

        pred_y = torch.stack(pred_y_list)
        pred_u = pred_y[:, :self.gene_dim]
        pred_s = pred_y[:, self.gene_dim:]
        return pred_u, pred_s
    
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