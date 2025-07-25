"""
Batch velocity encoder following VELOVI pattern for single velocity encoder per batch.

This implementation ensures all cells in a batch share the same ODE parameters,
following the pattern from the provided reference implementation.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torchode as to

from .base import SigmoidFeatureModule


class myVelocityEncoder(nn.Module):
    def __init__(self, num_genes: int):
        super().__init__()
        self.num_genes = num_genes
        self.W = nn.Linear(num_genes, num_genes)
        
    def forward(self, beta, gamma, interaction, c_open):
        def f(sigma, s, u):
            W_sigma = self.W(sigma)  # (batch_size, num_genes)
            du_dt = c_open * W_sigma - beta * u + interaction
            ds_dt = beta * u - gamma * s
            return torch.cat([du_dt, ds_dt], dim=1)
        return f


class myVelocityBatchWrapper(nn.Module):
    def __init__(self, num_genes: int, sigmoid_function: SigmoidFeatureModule):
        super().__init__()
        self.num_genes = num_genes
        self.sigmoid_function = sigmoid_function
    
    def forward(self, t: Union[float, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        u = y[:, :self.num_genes]  # (batch_size, num_genes)
        s = y[:, self.num_genes:2*self.num_genes]  # (batch_size, num_genes)
        
        dus_dt = self.f(self.sigmoid_function(s), s, u)
        
        # Reshape back to expected format
        return dus_dt


class VelocityEncoder(nn.Module):
    """
    Shared velocity encoder following VELOVI pattern.
    
    This class contains the shared ODE parameters and computes velocity
    for all genes/cells using the same dynamics equations. All batch samples
    must use the same parameters (beta, gamma, W, etc.) for the ODE solver.
    
    Args:
        num_genes: Number of genes.
        sigmoid_function: Pre-trained sigmoid module.
        W: Gene interaction weight matrix (shared across all cells).
    """
    
    def __init__(
        self,
        num_genes: int,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.W = nn.Linear(num_genes, num_genes)
        
        # These will be set for each batch - must be same for all cells in batch
        self.beta = None  # Shape: (num_genes,) - shared across batch
        self.gamma = None  # Shape: (num_genes,) - shared across batch
        self.interaction = None
        
    def set_shared_parameters(
        self,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        interaction: torch.Tensor
    ):
        """
        Set shared parameters for the batch.
        
        Following VELOVI pattern: all cells in a batch must use the same
        kinetic parameters for the ODE solver to work properly.
        
        Args:
            beta: Shared beta parameters (num_genes,).
            gamma: Shared gamma parameters (num_genes,).
        """
        self.beta = beta
        self.gamma = gamma
        self.interaction = interaction
        
    def forward(
        self,
        t: Union[float, torch.Tensor],
        sigma: torch.Tensor,
        u: torch.Tensor,
        s: torch.Tensor,
        c_open: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes velocity using shared parameters.
        
        Args:
            t: Time (unused in autonomous system).
            u: Unspliced counts (batch_size, num_genes).
            s: Spliced counts (batch_size, num_genes).
            c_open: Chromatin accessibility (batch_size, num_genes).
            interaction: Cell-specific interaction terms (batch_size, num_genes).
            
        Returns:
            Tuple of (du_dt, ds_dt, dc_open_dt).
        """
        if self.beta is None or self.gamma is None:
            raise RuntimeError("Must call set_shared_parameters before forward pass")
            
        # Gene interaction using shared W matrix
        W_sigma = torch.matmul(sigma, self.W.T)  # (batch_size, num_genes)
        
        # Velocity equations with shared parameters
        # Only interaction can be cell-specific, beta/gamma are shared
        du_dt = c_open * W_sigma - self.beta.unsqueeze(0) * u + self.interaction
        ds_dt = self.beta.unsqueeze(0) * u - self.gamma.unsqueeze(0) * s
    
        return du_dt, ds_dt


class VelocityBatchWrapper(nn.Module):
    """
    Batch wrapper for velocity encoder following VELOVI pattern.
    
    This reshapes inputs and outputs to work with torchode's expected format
    while using the shared velocity encoder. This wrapper ensures that the
    ODE solver can handle batch processing correctly.
    
    Args:
        velocity_encoder: The shared velocity encoder instance.
        num_genes: Number of genes.
    """
    
    def __init__(
        self,
        num_genes: int
    ):
        super().__init__()
        self.num_genes = num_genes

    def forward(self, t: Union[float, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the batch velocity encoder.
        
        Following VELOVI pattern where y contains the full state for all cells
        and genes, reshaped appropriately for the ODE solver.
        
        Args:
            t: Time (required for ODE solver interface).
            y: State tensor of shape (batch_size, 3*num_genes) representing
               [u, s, c_open] stacked for all cells.
               
        Returns:
            Velocity tensor of same shape as y.
        """
        
        # Reshape state: y is (batch_size, 3*num_genes)
        u = y[:, :self.num_genes]  # (batch_size, num_genes)
        s = y[:, self.num_genes:2*self.num_genes]  # (batch_size, num_genes)
        c_open = y[:, 2*self.num_genes:]  # (batch_size, num_genes)
        
        # Compute velocities
        du_dt, ds_dt = self.velocity_encoder(
            t, u, s, c_open, self.interaction
        )
        dc_open_dt = torch.zeros_like(c_open)  # Chromatin accessibility constant
        
        # Reshape back to expected format
        return torch.cat([du_dt, ds_dt, dc_open_dt], dim=1)


class BatchODESolver:
    """
    ODE solver that uses shared velocity encoder for batch processing.
    
    This class implements the VELOVI pattern where all cells in a batch
    share the same ODE parameters, ensuring efficient and correct simulation.
    """
    
    def __init__(
        self,
        velocity_encoder: VelocityEncoder,
        num_genes: int,
        dt0: torch.Tensor = None
    ):
        self.velocity_encoder = velocity_encoder
        self.num_genes = num_genes
        self.dt0 = dt0 if dt0 is not None else torch.ones(1)
        
        # Create batch wrapper
        self.batch_wrapper = VelocityBatchWrapper(velocity_encoder, num_genes)
        
    def solve_batch(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        interaction: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        c_open: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve ODE for a batch of cells using shared parameters.
        
        Args:
            t: Time points for each cell (batch_size, 1) or (batch_size,).
            x0: Initial conditions (batch_size, 3*num_genes).
            interaction: Cell-specific interactions (batch_size, num_genes).
            beta: Shared beta parameters (num_genes,).
            gamma: Shared gamma parameters (num_genes,).
            c_open: Shared chromatin accessibility parameters (num_genes,).
        Returns:
            Tuple of (pred_u, pred_s).
        """
        batch_size = x0.shape[0]
        
        # Set shared parameters - KEY: these are the same for all cells
        self.velocity_encoder.set_shared_parameters(beta, gamma)
        self.batch_wrapper.set_batch_interaction(interaction)
        
        # Set up ODE solver
        term = to.ODETerm(self.batch_wrapper)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        
        # Prepare time evaluation
        if t.dim() == 1:
            t = t.unsqueeze(1)  # Make sure it's (batch_size, 1)
            
        t0 = torch.zeros((batch_size, 1), device=x0.device)
        t_eval = torch.cat([t0, t], dim=1)  # (batch_size, 2)
        
        # Create and solve IVP
        ivp = to.InitialValueProblem(y0=x0, t_eval=t_eval)
        sol = solver.solve(ivp, dt0=self.dt0)
        
        # Extract results - sol.ys has shape (batch_size, time_steps, state_dim)
        final_state = sol.ys[:, -1]  # (batch_size, 3*num_genes)
        
        pred_u = final_state[:, :self.num_genes]
        pred_s = final_state[:, self.num_genes:2*self.num_genes]
        
        return pred_u, pred_s