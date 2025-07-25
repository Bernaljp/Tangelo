"""Base neural network components for Tangelo models."""

from typing import Optional, Tuple, Union, Type, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


def create_activation(name: Optional[str]) -> nn.Module:
    """
    Factory function to create an activation layer from its name.

    Args:
        name: Name of the activation function (e.g., 'relu', 'gelu').

    Returns:
        A PyTorch activation layer.
        
    Raises:
        NotImplementedError: If activation function is not implemented.
    """
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "prelu": nn.PReLU,
        "elu": nn.ELU,
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.2),
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        None: nn.Identity,
    }
    
    if name not in activation_map:
        raise NotImplementedError(f"Activation function '{name}' is not implemented.")
    
    return activation_map[name]()


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron with optional batch normalization, dropout, and residual connections.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Dimensionality of the hidden layers.
        output_dim: Dimensionality of the output.
        mlp_layers: Number of hidden layers.
        activation: Name of the activation function.
        bn: Whether to use batch normalization.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mlp_layers: int,
        activation: str = 'relu',
        bn: bool = False,
        dropout: float = 0.0,
        residual: bool = False
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        if bn:
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(create_activation(activation))
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(mlp_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if bn:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(create_activation(activation))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP."""
        identity = x
        for layer in self.layers:
            x = layer(x)
        
        # Simple residual connection if enabled and dimensions match
        if self.residual and x.shape == identity.shape:
            x = x + identity
        return x


class mySAGEConv(nn.Module):
    """
    A multi-layer GraphSAGE model.

    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden features.
        out_channels: Number of output features.
        n_layers: Number of GraphSAGE layers.
        activation_fn: Activation function name.
        batch_norm: Whether to use batch normalization.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        activation_fn: str = 'relu',
        batch_norm: bool = False,
        dropout: float = 0.0,
        residual: bool = False
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None
        self.activation = create_activation(activation_fn)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.n_layers = n_layers

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if self.bns is not None:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.bns is not None:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        if self.bns is not None and len(self.convs) > len(self.bns):
             self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GraphSAGE layers."""
        for i in range(self.n_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            
            if self.bns is not None:
                x = self.bns[i](x)
            
            # Apply activation to all but the last layer
            if i < self.n_layers - 1:
                x = self.activation(x)
            
            if self.residual and identity.shape == x.shape:
                x = x + identity
            
            x = self.dropout(x)
        return x


class SigmoidFeatureModule(nn.Module):
    """
    A PyTorch module that fits a double sigmoid function for each feature independently.
    The model predicts: ŷ = 0.5 * (sigmoid(a1*x + b1) + sigmoid(a2*x + b2))
    
    Args:
        num_features: Number of input features.
        init_a: Initial value for parameter 'a'.
        init_b: Initial value for parameter 'b'.
    """
    
    def __init__(
        self,
        num_features: int,
        init_a: float = 0.5,
        init_b: float = 5.0
    ) -> None:
        super().__init__()
        self.a1 = nn.Parameter(torch.full((num_features,), init_a))
        self.a2 = nn.Parameter(torch.full((num_features,), 2 * init_a))
        self.b1 = nn.Parameter(torch.full((num_features,), init_b))
        self.b2 = nn.Parameter(torch.full((num_features,), init_b / 2))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the double sigmoid transformation."""
        return 0.5 * (
            torch.sigmoid(self.a1 * x + self.b1) + 
            torch.sigmoid(self.a2 * x + self.b2)
        )
    
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the current values of the parameters."""
        return (
            self.a1.data.clone(),
            self.a2.data.clone(), 
            self.b1.data.clone(),
            self.b2.data.clone()
        )
    
    def set_parameters(
        self,
        params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Sets the parameters from a given tuple."""
        a1_val, a2_val, b1_val, b2_val = params
        with torch.no_grad():
            self.a1.data.copy_(a1_val)
            self.a2.data.copy_(a2_val)
            self.b1.data.copy_(b1_val)
            self.b2.data.copy_(b2_val)


class SigmoidFeatureTrainer:
    """
    A trainer class for the SigmoidFeatureModule.
    
    Args:
        model: The model to train.
        learning_rate: The learning rate for the optimizer.
        optimizer_class: The PyTorch optimizer class to use.
        criterion: The loss function.
        scheduler_class: The learning rate scheduler class.
        scheduler_kwargs: Arguments for the scheduler.
    """
    
    def __init__(
        self,
        model: SigmoidFeatureModule,
        learning_rate: float = 0.01,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        criterion: nn.Module = nn.MSELoss(),
        scheduler_class: Optional[Type[_LRScheduler]] = torch.optim.lr_scheduler.StepLR,
        scheduler_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.model = model
        self.optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        self.criterion = criterion
        
        if scheduler_class:
            scheduler_kwargs = scheduler_kwargs or {'step_size': 100, 'gamma': 0.1}
            self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)
        else:
            self.scheduler = None

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Performs a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1000,
        verbose: bool = True
    ) -> list[float]:
        """Fits the model to the data."""
        loss_history = []
        pbar = tqdm(range(epochs), desc="Pre-training Sigmoid", disable=not verbose)
        
        for epoch in pbar:
            loss = self.train_step(x, y)
            loss_history.append(loss)
            
            if self.scheduler:
                self.scheduler.step()
            
            if verbose and (epoch + 1) % 100 == 0:
                pbar.set_postfix({'Loss': f'{loss:.6f}'})
                
        return loss_history


class VelocityModel(nn.Module):
    """
    Defines the ODE system for RNA velocity. This is a lightweight, single-use
    class that is initialized with all necessary parameters for one simulation.
    The state vector `y` is a concatenation of [u, s].

    Args:
        num_genes: Number of genes.
        sigmoid_function: The pre-trained sigmoid module.
        W_effective: The pre-computed effective weight matrix.
        interaction: The interaction parameter for the simulation.
        beta: The beta parameter for the simulation.
        gamma: The gamma parameter for the simulation.
        c: The c parameter for modulating sigma.
    """
    
    def __init__(
        self,
        num_genes: int,
        sigmoid_function: SigmoidFeatureModule,
        W_effective: torch.Tensor,
        interaction: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        c: torch.Tensor
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.sigmoid = sigmoid_function
        self.W_effective = W_effective
        self.interaction = interaction
        self.beta = beta
        self.gamma = gamma
        self.c = c

    def forward(self, t: Union[float, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Computes the velocity vector dy/dt.

        Args:
            t: Current time (often unused in autonomous systems).
            y: State tensor of shape (batch, 2*gene_dim), representing [u, s].

        Returns:
            The velocity vector dy/dt.
        """
        u = y[:, :self.num_genes]
        s = y[:, self.num_genes:]
        
        # Modulate sigma with the new parameter c
        sigma = self.sigmoid(s)

        # Perform batch matrix-vector product with the effective weight matrix
        Ws_total = torch.bmm(self.W_effective, sigma.unsqueeze(-1)).squeeze(-1)

        du_dt = self.c * Ws_total - self.beta * u + self.interaction
        ds_dt = self.beta * u - self.gamma * s
    
        return torch.cat([du_dt, ds_dt], dim=1)