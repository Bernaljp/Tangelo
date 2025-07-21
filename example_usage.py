"""
Example usage of the Tangelo package.

This script demonstrates how to use the Tangelo package to replace
the functionality in the original Jupyter notebook.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tangelo as tg

def main():
    """Main example workflow."""
    
    # Configuration
    data_path = "/path/to/your/data/SRR28572641/"
    
    # 1. Setup multi-modal data (replaces notebook cells 5-17)
    print("Setting up multi-modal data...")
    adata = tg.setup_multimodal_data(
        data_path=data_path,
        use_mm10=True
    )
    
    # Preprocess multi-modal data (replaces notebook preprocessing)
    import dynamo as dyn
    import muon as mu
    
    # Setup ATAC preprocessing
    mu.atac.tl.locate_fragments(adata, data_path + 'atac/GSM8189706_ME13_50um_3_ATAC.fragments.tsv.gz')
    
    # Process RNA velocity
    dyn.tl.moments(adata.mod['rna'])
    
    # Intersect observations
    mu.pp.intersect_obs(adata)
    
    print(f"Data shape: {adata.n_obs} cells × {adata.n_vars} features")
    
    # 2. Create configuration (replaces notebook cells 47-48)
    config = tg.TangeloConfig(
        gene_dim=adata['rna'].n_vars,
        atac_dim=adata['atac'].n_vars,
        spatial_dim=2,
        latent_dim=10,
        n_components=3,
        hidden_dim_expression=128,
        hidden_dim_spatial=128,
        hidden_dim_decoder=128,
        gnn_layers=2,
        mlp_layers=2,
        n_neighbors_expression=30,
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=128,
        tangent_loss_kwargs={'a': 1.0, 'b': 2.0, 'lambda_reg': 0.1}
    )
    
    # Save configuration for reproducibility
    tg.save_config(config, "tangelo_config.yaml")
    
    # 3. Initialize model (replaces notebook cell 49)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Train model (replaces notebook cell 50)
    trainer = tg.TangeloTrainer(model, learning_rate=config.learning_rate, device=device)
    
    print("Starting training...")
    loss_history = trainer.train(
        adata,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        num_neighbors=[10, 5],
        knn_use_unspliced=False,
        sigmoid_epochs=1000,
        sigmoid_lr=1.0
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Create visualizations (replaces notebook cells 19-25)
    print("Creating visualizations...")
    
    # Create UMAP embeddings for different data types
    spliced_colors = tg.create_rgb_colors(
        adata['rna'].layers['M_s'], 
        method='umap', 
        n_neighbors=30, 
        min_dist=1.0
    )
    
    ms_colors = tg.create_rgb_colors(
        adata['rna'].layers['M_s'],
        method='umap',
        n_neighbors=15,
        min_dist=0.1
    )
    
    atac_colors = tg.create_rgb_colors(
        adata['atac'].layers['counts'],
        method='umap',
        n_neighbors=15,
        min_dist=1.0
    )
    
    # Load background image if available
    img_path = data_path + 'spatial/tissue_lowres_image.png'
    try:
        img = np.fliplr(mpimg.imread(img_path))
    except FileNotFoundError:
        print(f"Background image not found at {img_path}")
        img = np.ones((2174, 2174, 3))  # White background
    
    x_pixels = adata.obs['x_pixel'].values.astype(int)
    y_pixels = adata.obs['y_pixel'].values.astype(int)
    
    # Plot spliced RNA UMAP
    fig1 = tg.plot_umap_components_and_rgb(
        spliced_colors, x_pixels, y_pixels, img,
        spatial_shape=(50, 50),
        extent=(0, 2174, 0, 2174),
        titles=('RNA Spliced 1', 'RNA Spliced 2', 'RNA Spliced 3', 'Combined RGB')
    )
    fig1.savefig('rna_spliced_umap.png', dpi=300, bbox_inches='tight')
    
    # Plot moment-smoothed RNA UMAP  
    fig2 = tg.plot_umap_components_and_rgb(
        ms_colors, x_pixels, y_pixels, img,
        titles=('RNA M_s 1', 'RNA M_s 2', 'RNA M_s 3', 'Combined RGB')
    )
    fig2.savefig('rna_ms_umap.png', dpi=300, bbox_inches='tight')
    
    # Plot ATAC UMAP
    fig3 = tg.plot_umap_components_and_rgb(
        atac_colors, x_pixels, y_pixels, img,
        titles=('ATAC 1', 'ATAC 2', 'ATAC 3', 'Combined RGB')
    )
    fig3.savefig('atac_umap.png', dpi=300, bbox_inches='tight')
    
    # 6. Evaluate model
    print("Evaluating model...")
    eval_metrics = trainer.evaluate(adata, batch_size=64)
    print(f"Evaluation loss: {eval_metrics['loss']:.4f}")
    
    # 7. Save model
    trainer.save_model('tangelo_model.pth')
    
    print("Training and visualization complete!")
    print("Generated files:")
    print("- tangelo_config.yaml: Configuration file")
    print("- tangelo_model.pth: Trained model checkpoint")
    print("- training_loss.png: Training loss plot")
    print("- rna_spliced_umap.png: RNA spliced UMAP visualization")
    print("- rna_ms_umap.png: RNA moment-smoothed UMAP visualization") 
    print("- atac_umap.png: ATAC UMAP visualization")


def small_dataset_example():
    """Example with a smaller dataset for testing."""
    
    print("Running small dataset example...")
    
    # This would replace notebook cells 46, 56-60
    # Create a smaller subset for faster testing
    
    data_path = "/path/to/your/data/SRR28572641/"
    
    # Load full dataset
    adata = tg.setup_multimodal_data(data_path, use_mm10=True)
    
    # Create small subset
    import muon as mu
    np.random.seed(42)
    randn_cells = np.random.choice(adata.n_obs, 200, replace=False)
    randn_genes = np.random.choice(adata.mod['rna'].n_vars, 500, replace=False)
    randn_atac = np.random.choice(adata.mod['atac'].n_vars, 5000, replace=False)
    
    small_adata = mu.MuData({
        'atac': adata.mod['atac'][randn_cells][:, randn_atac],
        'rna': adata.mod['rna'][randn_cells][:, randn_genes]
    })
    small_adata.obs = adata.obs.iloc[randn_cells]
    
    # Quick training configuration
    config = tg.TangeloConfig(
        gene_dim=small_adata['rna'].n_vars,
        atac_dim=small_adata['atac'].n_vars,
        spatial_dim=2,
        latent_dim=10,
        n_components=3,
        learning_rate=1e-3,
        n_epochs=2,  # Very short training for testing
        batch_size=10
    )
    
    # Initialize and train model
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
    
    trainer = tg.TangeloTrainer(model, learning_rate=config.learning_rate)
    loss_history = trainer.train(small_adata, n_epochs=2, batch_size=10)
    
    print(f"Small dataset training completed. Final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run small dataset example
    # small_dataset_example()