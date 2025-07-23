"""Simplified trainer for the new Tangelo architecture."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch_geometric.utils
from tqdm import tqdm
import muon as mu
import scipy as scp

from ..models.simplified import SimplifiedTangeloModel
from ..data.simplified_preprocessing import (
    setup_simplified_data, 
    create_simplified_batch_data,
    validate_simplified_data,
    compare_data_sizes
)


class SimplifiedTangeloTrainer:
    """
    Trainer class for the simplified Tangelo model.
    
    Key differences from original trainer:
    1. No ATAC peaks in data preparation
    2. Population-level ODE simulation
    3. Simplified data loading workflow
    4. c_open handled as separate state variable
    
    Args:
        model: The simplified Tangelo model to train.
        learning_rate: Learning rate for optimization.
        device: Device to run training on.
    """
    
    def __init__(
        self,
        model: SimplifiedTangeloModel,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None
    ) -> None:
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def prepare_data(
        self,
        adata: mu.MuData,
        n_neighbors_spatial: int = 8,
        n_neighbors_expression: int = 30,
    ) -> Tuple[scp.sparse.csr_matrix, scp.sparse.csr_matrix, torch.Tensor, torch.Tensor]:
        """
        Prepares simplified data for training.
        
        Args:
            adata: MuData object containing multi-modal data.
            n_neighbors_spatial: Number of spatial neighbors for graph.
            n_neighbors_expression: Number of expression neighbors for graph.
            
        Returns:
            Tuple of (input_matrix, c_open_matrix, spatial_edge_index, expression_edge_index).
        """
        print("Preparing simplified data for training...")
        
        # Validate that we have the required data
        assert 'rna' in adata.mod, "RNA modality not found in data"
        assert 'M_u' in adata['rna'].layers, "Unspliced layer (M_u) not found"
        assert 'M_s' in adata['rna'].layers, "Spliced layer (M_s) not found"
        assert 'open_chromatin' in adata['rna'].layers, "Open chromatin layer not found"
        
        # Show data size comparison
        if 'atac' in adata.mod:
            print("Comparing data sizes...")
            # We'll compute this after setup
        
        # Setup simplified data
        X, C_open, spatial_edge_index, expression_edge_index = setup_simplified_data(
            adata, n_neighbors_spatial, n_neighbors_expression
        )
        
        # Validate data format
        validate_simplified_data(X, C_open, self.model.gene_dim)
        
        # Show size comparison if ATAC data was present
        if 'atac' in adata.mod:
            compare_data_sizes(adata, X)
        
        # Store distance matrix for tangent loss (simplified version)
        expr_data = adata['rna'].layers['M_s'].toarray() if hasattr(adata['rna'].layers['M_s'], 'toarray') else adata['rna'].layers['M_s']
        expr_tensor = torch.tensor(expr_data, dtype=torch.float32)
        dist_matrix = torch.cdist(expr_tensor, expr_tensor, p=2)
        
        flat_values = dist_matrix[expression_edge_index[0], expression_edge_index[1]].flatten()
        self.model.dist_matrix = torch.sparse_coo_tensor(
            expression_edge_index,
            flat_values,
            (adata.n_obs, adata.n_obs)
        ).to(self.device)
        
        return (
            X,
            C_open,
            spatial_edge_index.to(self.device),
            expression_edge_index.to(self.device)
        )
    
    def train(
        self,
        adata: mu.MuData,
        n_epochs: int = 100,
        batch_size: int = 128,
        num_neighbors: List[int] = [10, 5],
        n_neighbors_spatial: int = 8,
        n_neighbors_expression: int = 30,
        knn_use_unspliced: bool = False,
        sigmoid_epochs: int = 1000,
        sigmoid_lr: float = 1.0
    ) -> List[float]:
        """
        Main training loop for the simplified Tangelo model.
        
        Args:
            adata: MuData object containing the training data.
            n_epochs: Number of training epochs.
            batch_size: Batch size for training.
            num_neighbors: Number of neighbors for NeighborLoader.
            n_neighbors_spatial: Number of spatial neighbors for graph.
            n_neighbors_expression: Number of expression neighbors for graph.
            sigmoid_epochs: Number of epochs for sigmoid pre-training.
            sigmoid_lr: Learning rate for sigmoid pre-training.
            
        Returns:
            List of training losses.
        """
        # Pre-train sigmoid function (same as before)
        print("Pre-training sigmoid function...")
        s_data = adata['rna'].layers['M_s'].toarray().astype(np.float32)
        s_tensor = torch.tensor(s_data, device=self.device)
        self.model.pretrain_sigmoid(s_tensor, sigmoid_epochs, sigmoid_lr)
        
        # Prepare simplified data
        X, C_open, spatial_edge_index, expression_edge_index = self.prepare_data(
            adata, n_neighbors_spatial, n_neighbors_expression
        )
        
        # Create data loader
        full_data = Data(edge_index=expression_edge_index, num_nodes=X.shape[0])
        train_loader = NeighborLoader(
            full_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True,
        )
        
        # Training loop
        self.model.train()
        loss_history = []
        
        for epoch in range(n_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            epoch_losses = []
            
            for batch in pbar:
                loss = self._train_step(
                    batch, X, C_open, spatial_edge_index, expression_edge_index
                )
                epoch_losses.append(loss)
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        return loss_history
    
    def _train_step(
        self,
        batch: Data,
        X: scp.sparse.csr_matrix,
        C_open: scp.sparse.csr_matrix,
        spatial_edge_index: torch.Tensor,
        expression_edge_index: torch.Tensor
    ) -> float:
        """Performs a single training step with simplified data."""
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        idxs = batch.n_id.cpu().numpy()
        
        # Create batch data using simplified preprocessing
        batch_input, c_open_batch, u_batch, s_batch = create_simplified_batch_data(
            X, C_open, idxs, self.model.gene_dim, self.device
        )
        
        # Create batch spatial edge index
        batch_space_edge_index, _ = torch_geometric.utils.subgraph(
            batch.n_id, spatial_edge_index, relabel_nodes=True
        )
        
        # Forward pass
        pred_u, pred_s, qz_mean, qz_log_var = self.model(
            batch_input, c_open_batch, batch_space_edge_index, batch.edge_index
        )
        
        # Compute loss
        loss = self.model.loss(u_batch, s_batch, pred_u, pred_s, qz_mean, qz_log_var, batch.edge_index)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        adata: mu.MuData,
        batch_size: int = 128,
        num_neighbors: List[int] = [10, 5]
    ) -> Dict[str, float]:
        """
        Evaluates the simplified model on the given data.
        
        Args:
            adata: MuData object containing the evaluation data.
            batch_size: Batch size for evaluation.
            num_neighbors: Number of neighbors for NeighborLoader.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        
        # Prepare data (reuse training preparation)
        X, C_open, spatial_edge_index, expression_edge_index = self.prepare_data(adata)
        
        # Create data loader
        full_data = Data(edge_index=expression_edge_index, num_nodes=X.shape[0])
        eval_loader = NeighborLoader(
            full_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
        )
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                idxs = batch.n_id.cpu().numpy()
                
                # Create batch data
                batch_input, c_open_batch, u_batch, s_batch = create_simplified_batch_data(
                    X, C_open, idxs, self.model.gene_dim, self.device
                )
                
                # Create batch spatial edge index
                batch_space_edge_index, _ = torch_geometric.utils.subgraph(
                    batch.n_id, spatial_edge_index, relabel_nodes=True
                )
                
                # Forward pass
                pred_u, pred_s, qz_mean, qz_log_var = self.model(
                    batch_input, c_open_batch, batch_space_edge_index, batch.edge_index
                )
                
                # Compute loss
                loss = self.model.loss(u_batch, s_batch, pred_u, pred_s, qz_mean, qz_log_var, batch.edge_index)
                
                total_loss += loss.item() * len(batch.n_id)
                total_samples += len(batch.n_id)
        
        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}
    
    def save_model(self, path: str) -> None:
        """Saves the model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'gene_dim': self.model.gene_dim,
                'spatial_dim': self.model.spatial_dim,
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
            }
        }, path)
        print(f"Simplified model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Loads the model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Simplified model loaded from {path}")
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"Model config: {config}")


def create_simplified_model_from_config(
    gene_dim: int,
    spatial_dim: int = 2,
    **kwargs
) -> SimplifiedTangeloModel:
    """
    Creates a simplified Tangelo model with reasonable defaults.
    
    Args:
        gene_dim: Number of genes.
        spatial_dim: Number of spatial dimensions.
        **kwargs: Additional model parameters.
        
    Returns:
        Initialized SimplifiedTangeloModel.
    """
    default_config = {
        'activation_fn': 'relu',
        'batch_norm': False,
        'dropout': 0.0,
        'residual': False,
        'hidden_dim_expression': 128,
        'hidden_dim_spatial': 128,
        'hidden_dim_decoder': 128,
        'latent_dim': 10,
        'gnn_layers': 2,
        'mlp_layers': 2,
        'n_neighbors': 30,
        'tangent_loss_kwargs': {'lambda_reg': 0.1}
    }
    
    # Update with user-provided kwargs
    default_config.update(kwargs)
    
    return SimplifiedTangeloModel(
        gene_dim=gene_dim,
        spatial_dim=spatial_dim,
        **default_config
    )