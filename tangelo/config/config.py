"""Configuration management for Tangelo models."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import yaml
from pathlib import Path


@dataclass
class TangeloConfig:
    """
    Configuration class for Tangelo models.
    
    Model Architecture:
        gene_dim: Number of genes.
        atac_dim: Number of ATAC features.
        spatial_dim: Number of spatial dimensions.
        latent_dim: Latent space dimension.
        n_components: Number of mixture components.
        hidden_dim_expression: Hidden dimension for expression branch.
        hidden_dim_spatial: Hidden dimension for spatial branch.
        hidden_dim_decoder: Hidden dimension for decoders.
        gnn_layers: Number of GNN layers.
        mlp_layers: Number of MLP layers.
        
    Training:
        learning_rate: Learning rate for optimization.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        sigmoid_epochs: Number of epochs for sigmoid pre-training.
        sigmoid_lr: Learning rate for sigmoid pre-training.
        
    Graph Construction:
        n_neighbors_spatial: Number of spatial neighbors.
        n_neighbors_expression: Number of expression neighbors.
        num_neighbors: Number of neighbors for NeighborLoader.
        
    Model Options:
        activation_fn: Activation function name.
        batch_norm: Whether to use batch normalization.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
        knn_use_unspliced: Whether to use unspliced data for k-NN.
        
    Loss Configuration:
        tangent_loss_kwargs: Arguments for tangent loss computation.
    """
    
    # Model architecture
    gene_dim: int = 500
    atac_dim: int = 5000
    spatial_dim: int = 2
    latent_dim: int = 10
    n_components: int = 3
    hidden_dim_expression: int = 128
    hidden_dim_spatial: int = 128
    hidden_dim_decoder: int = 128
    gnn_layers: int = 2
    mlp_layers: int = 2
    
    # Training parameters
    learning_rate: float = 1e-3
    n_epochs: int = 100
    batch_size: int = 128
    sigmoid_epochs: int = 1000
    sigmoid_lr: float = 1.0
    
    # Graph construction
    n_neighbors_spatial: int = 8
    n_neighbors_expression: int = 30
    num_neighbors: list = None
    
    # Model options
    activation_fn: str = 'relu'
    batch_norm: bool = False
    dropout: float = 0.0
    residual: bool = False
    knn_use_unspliced: bool = False
    
    # Loss configuration
    tangent_loss_kwargs: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.num_neighbors is None:
            self.num_neighbors = [10, 5]
        if self.tangent_loss_kwargs is None:
            self.tangent_loss_kwargs = {'a': 1.0, 'b': 2.0, 'lambda_reg': 0.1}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TangeloConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'TangeloConfig':
        """Update configuration with new values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


def load_config(config_path: str) -> TangeloConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML).
        
    Returns:
        TangeloConfig object.
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix.lower() == '.json':
            config_dict = json.load(f)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    return TangeloConfig.from_dict(config_dict)


def save_config(config: TangeloConfig, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: TangeloConfig object to save.
        config_path: Path where to save the configuration.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(path, 'w') as f:
        if path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    print(f"Configuration saved to {config_path}")