# Tangelo: Deep Learning for Multi-Modal Single-Cell Analysis

Tangelo is a PyTorch package for integrating spatial transcriptomics, RNA velocity, and ATAC-seq data using graph neural networks and ordinary differential equation modeling.

## Features

- **Multi-modal Integration**: Seamlessly combine RNA-seq, ATAC-seq, and spatial transcriptomics data
- **Graph Neural Networks**: Leverage spatial and expression relationships through GraphSAGE architectures  
- **RNA Velocity Modeling**: ODE-based simulation of RNA dynamics with torchode
- **Spatial Visualization**: Rich plotting tools for spatial gene expression and velocity fields
- **Flexible Configuration**: YAML/JSON-based configuration management
- **Type Safety**: Comprehensive type hints throughout the codebase

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tangelo.git
cd tangelo/tangelo_package

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,docs]"
```

## Quick Start

```python
import tangelo as tg
import muon as mu

# Load and setup multi-modal data
adata = tg.setup_multimodal_data(
    data_path="/path/to/data",
    use_mm10=True
)

# Create configuration
config = tg.TangeloConfig(
    gene_dim=500,
    atac_dim=5000,
    latent_dim=10,
    n_epochs=100
)

# Initialize model
model = tg.TangeloModel(
    gene_dim=config.gene_dim,
    atac_dim=config.atac_dim,
    spatial_dim=config.spatial_dim,
    activation_fn=config.activation_fn,
    batch_norm=config.batch_norm,
    dropout=config.dropout,
    residual=config.residual,
    hidden_dim_expression=config.hidden_dim_expression,
    hidden_dim_spatial=config.hidden_dim_spatial,
    hidden_dim_decoder=config.hidden_dim_decoder,
    latent_dim=config.latent_dim,
    n_components=config.n_components,
    gnn_layers=config.gnn_layers,
    mlp_layers=config.mlp_layers,
    n_neighbors=config.n_neighbors_expression,
    tangent_loss_kwargs=config.tangent_loss_kwargs
)

# Train model
trainer = tg.TangeloTrainer(model, learning_rate=config.learning_rate)
loss_history = trainer.train(adata, n_epochs=config.n_epochs)

# Visualize results
colors = tg.create_umap_embedding(adata['rna'].layers['M_s'])
tg.plot_umap_components_and_rgb(
    colors, adata.obs['x_pixel'], adata.obs['y_pixel'], img
)
```

## Package Structure

```
tangelo/
├── models/           # Core neural network models
│   ├── base.py      # Basic building blocks (MLP, GraphSAGE, etc.)
│   ├── encoder.py   # Variational autoencoder components
│   └── tangelo.py   # Main Tangelo model
├── data/            # Data processing utilities
│   ├── preprocessing.py  # Multi-modal data setup
│   └── utils.py     # Utility functions (CDF calculation, etc.)
├── training/        # Training and optimization
│   └── trainer.py   # TangeloTrainer class
├── config/          # Configuration management
│   └── config.py    # TangeloConfig dataclass
└── visualization/   # Plotting and visualization
    ├── spatial.py   # Spatial plotting functions
    └── utils.py     # UMAP/PCA embedding utilities
```

## Key Components

### Models
- **TangeloModel**: Main model integrating VAE, GNN, and ODE components
- **GraphVAEncoder**: Dual-branch variational autoencoder for spatial and expression data
- **VelocityModel**: ODE system for RNA velocity simulation
- **SigmoidFeatureModule**: Learnable sigmoid functions for CDF modeling

### Data Processing
- **setup_multimodal_data**: Load and integrate multi-modal datasets
- **create_peak_by_cell_matrix**: Process ATAC-seq fragment files
- **create_graph_data**: Build k-NN graphs for spatial and expression data

### Training
- **TangeloTrainer**: High-level training interface with preprocessing and evaluation
- Automatic graph construction and batch processing
- Model checkpointing and evaluation metrics

### Visualization
- **plot_umap_components_and_rgb**: RGB spatial visualization from UMAP
- **plot_spatial_gene_expression**: Gene expression overlays on tissue images
- **create_umap_embedding**: Dimensionality reduction utilities

## Configuration

Use YAML or JSON files for reproducible experiments:

```yaml
# config.yaml
gene_dim: 500
atac_dim: 5000
spatial_dim: 2
latent_dim: 10
n_components: 3
learning_rate: 0.001
n_epochs: 100
batch_size: 128
tangent_loss_kwargs:
  a: 1.0
  b: 2.0
  lambda_reg: 0.1
```

```python
config = tg.load_config("config.yaml")
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12.0  
- torch-geometric ≥ 2.0.0
- torchode ≥ 0.2.0
- scanpy, muon, dynamo-release
- numpy, scipy, pandas, scikit-learn
- matplotlib, umap-learn

## License

MIT License - see LICENSE file for details.

## Citation

If you use Tangelo in your research, please cite:

```bibtex
@software{tangelo2024,
  title={Tangelo: Deep Learning for Multi-Modal Single-Cell Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tangelo}
}
```