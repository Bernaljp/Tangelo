"""Training module for Tangelo models."""

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

from ..models.tangelo import TangeloModel
from ..data.preprocessing import create_graph_data


class TangeloTrainer:
    """
    Trainer class for Tangelo models.
    
    Args:
        model: The Tangelo model to train.
        learning_rate: Learning rate for optimization.
        device: Device to run training on.
    """
    
    def __init__(
        self,
        model: TangeloModel,
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
        knn_use_unspliced: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares data for training by creating graphs and organizing features.
        
        Args:
            adata: MuData object containing multi-modal data.
            n_neighbors_spatial: Number of spatial neighbors for graph.
            n_neighbors_expression: Number of expression neighbors for graph.
            knn_use_unspliced: Whether to use unspliced data for expression graph.
            
        Returns:
            Tuple of (input_matrix, spatial_edge_index, expression_edge_index, dist_matrix).
        """
        print("Computing k-NN graphs...")
        
        # Create graphs
        spatial_edge_index, expression_edge_index, dist_matrix = create_graph_data(
            adata, n_neighbors_spatial, n_neighbors_expression, knn_use_unspliced
        )
        
        # Prepare input matrix: [U, S, C_open, C, spatial_coordinates]
        S = adata['rna'].layers['M_s']
        U = adata['rna'].layers['M_u']
        C_open = adata['rna'].layers['open_chromatin']
        C = adata['atac'].layers['counts']
        spatial_coords = scp.sparse.csr_matrix(
            adata.obs[['x_position', 'y_position']].values
        )
        
        X = scp.sparse.hstack([S, U, C_open, C, spatial_coords]).tocsr()
        
        # Store distance matrix in model
        flat_values = dist_matrix[expression_edge_index[0], expression_edge_index[1]].flatten()
        self.model.dist_matrix = torch.sparse_coo_tensor(
            expression_edge_index,
            flat_values,
            (adata.n_obs, adata.n_obs)
        ).to(self.device)
        
        return (
            X,
            spatial_edge_index.to(self.device),
            expression_edge_index.to(self.device),
            dist_matrix.to(self.device)
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
        Main training loop for the Tangelo model.
        
        Args:
            adata: MuData object containing the training data.
            n_epochs: Number of training epochs.
            batch_size: Batch size for training.
            num_neighbors: Number of neighbors for NeighborLoader.
            n_neighbors_spatial: Number of spatial neighbors for graph.
            n_neighbors_expression: Number of expression neighbors for graph.
            knn_use_unspliced: Whether to use unspliced data for k-NN.
            sigmoid_epochs: Number of epochs for sigmoid pre-training.
            sigmoid_lr: Learning rate for sigmoid pre-training.
            
        Returns:
            List of training losses.
        """
        # Pre-train sigmoid function
        print("Pre-training sigmoid function...")
        if knn_use_unspliced:
            s_data = adata['rna'].layers['M_u'].toarray().astype(np.float32)
        else:
            s_data = adata['rna'].layers['M_s'].toarray().astype(np.float32)
        
        s_tensor = torch.tensor(s_data, device=self.device)
        self.model.pretrain_sigmoid(s_tensor, sigmoid_epochs, sigmoid_lr)
        
        # Prepare data
        X, spatial_edge_index, expression_edge_index, _ = self.prepare_data(
            adata, n_neighbors_spatial, n_neighbors_expression, knn_use_unspliced
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
                    batch, X, spatial_edge_index, expression_edge_index, adata
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
        spatial_edge_index: torch.Tensor,
        expression_edge_index: torch.Tensor,
        adata: mu.MuData
    ) -> float:
        """Performs a single training step."""
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        idxs = batch.n_id.cpu().numpy()
        batch_x = torch.tensor(X[idxs].toarray().astype(np.float32), device=self.device)
        
        # Extract components
        u = batch_x[:, :self.model.gene_dim]
        s = batch_x[:, self.model.gene_dim:2*self.model.gene_dim]
        c_open = batch_x[:, 2*self.model.gene_dim:3*self.model.gene_dim]
        
        # Create batch spatial edge index
        batch_space_edge_index, _ = torch_geometric.utils.subgraph(
            batch.n_id, spatial_edge_index, relabel_nodes=True
        )
        
        # Prepare input: [u, s, c, spatial_coords]
        batch_input = torch.cat([
            batch_x[:, :2*self.model.gene_dim],  # u and s
            batch_x[:, 3*self.model.gene_dim:],  # c and spatial_coordinates
        ], dim=1)
        
        # Forward pass
        pred_u, pred_s, qz_mean, qz_log_var = self.model(
            batch_input, c_open, batch_space_edge_index, batch.edge_index
        )
        
        # Compute loss
        loss = self.model.loss(u, s, pred_u, pred_s, qz_mean, qz_log_var, batch.edge_index)
        
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
        Evaluates the model on the given data.
        
        Args:
            adata: MuData object containing the evaluation data.
            batch_size: Batch size for evaluation.
            num_neighbors: Number of neighbors for NeighborLoader.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        
        # Prepare data (reuse training preparation)
        X, spatial_edge_index, expression_edge_index, _ = self.prepare_data(adata)
        
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
                batch_x = torch.tensor(X[idxs].toarray().astype(np.float32), device=self.device)
                
                # Extract components
                u = batch_x[:, :self.model.gene_dim]
                s = batch_x[:, self.model.gene_dim:2*self.model.gene_dim]
                c_open = batch_x[:, 2*self.model.gene_dim:3*self.model.gene_dim]
                
                # Create batch spatial edge index
                batch_space_edge_index, _ = torch_geometric.utils.subgraph(
                    batch.n_id, spatial_edge_index, relabel_nodes=True
                )
                
                # Prepare input
                batch_input = torch.cat([
                    batch_x[:, :2*self.model.gene_dim],
                    batch_x[:, 3*self.model.gene_dim:],
                ], dim=1)
                
                # Forward pass
                pred_u, pred_s, qz_mean, qz_log_var = self.model(
                    batch_input, c_open, batch_space_edge_index, batch.edge_index
                )
                
                # Compute loss
                loss = self.model.loss(u, s, pred_u, pred_s, qz_mean, qz_log_var, batch.edge_index)
                
                total_loss += loss.item() * len(batch.n_id)
                total_samples += len(batch.n_id)
        
        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}
    
    def save_model(self, path: str) -> None:
        """Saves the model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Loads the model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")