"""Data preprocessing functions for Tangelo."""

from typing import Tuple, Optional, Union
import os
import numpy as np
import pandas as pd
import pysam
import torch
import scipy as scp
from scipy.sparse import dok_matrix, csr_matrix
from scipy.io import mmwrite
from sklearn.neighbors import kneighbors_graph
import muon as mu
import dynamo as dyn


def create_peak_by_cell_matrix(
    data_path: str,
    name: str,
    redo: bool = False
) -> None:
    """
    Generates a peak-by-cell matrix from ATAC-seq fragment and peak files.

    This function checks if the output files already exist and will skip
    computation unless the 'redo' flag is set to True.

    Args:
        data_path: The path to the directory containing the input files.
        name: The base name of the sample files.
        redo: If True, re-runs the analysis even if output files exist.
    """
    # Define file paths
    input_peak_file = os.path.join(data_path, f'{name}_ATAC_peaks.narrowPeak')
    input_fragment_file = os.path.join(data_path, f'{name}.fragments.tsv.gz')
    
    output_dir = os.path.join(data_path, "peak_by_cell_matrix")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_matrix_file = os.path.join(output_dir, "matrix.mtx")
    output_peaks_file = os.path.join(output_dir, "features.tsv")
    output_barcodes_file = os.path.join(output_dir, "barcodes.tsv")
    
    # Check if output already exists
    output_exists = (
        all(os.path.exists(f) for f in [output_matrix_file, output_peaks_file, output_barcodes_file]) or 
        all(os.path.exists(f+'.gz') for f in [output_matrix_file, output_peaks_file, output_barcodes_file])
    )
    
    if output_exists and not redo:
        print(f"✅ Output files already exist in '{output_dir}'. Skipping.")
        print("Set redo=True to re-generate.")
        return

    print(f"Processing sample: {name}")

    # Load peaks
    print("Step 1: Loading peaks...")
    peak_cols = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
                 "signalValue", "pValue", "qValue", "peak"]
    peaks_df = pd.read_csv(input_peak_file, sep='\t', header=None, names=peak_cols)
    peaks_df['peak_id'] = (peaks_df['chrom'] + ':' + 
                          peaks_df['chromStart'].astype(str) + '-' + 
                          peaks_df['chromEnd'].astype(str))
    print(f"-> Found {len(peaks_df)} peaks.")

    # Get unique cell barcodes
    print("\nStep 2: Getting all unique cell barcodes...")
    all_barcodes = set()
    with pysam.TabixFile(input_fragment_file, 'r') as tabix:
        for row in tabix.fetch(parser=pysam.asTuple()):
            all_barcodes.add(row[3])  # Barcode is in the 4th column

    barcodes = sorted(list(all_barcodes))
    barcode_map = {barcode: i for i, barcode in enumerate(barcodes)}
    print(f"-> Found {len(barcodes)} unique cell barcodes.")

    # Build the peak-by-cell matrix
    print("\nStep 3: Building the peak-by-cell matrix...")
    matrix = dok_matrix((len(peaks_df), len(barcodes)), dtype=int)
    tabixfile = pysam.TabixFile(input_fragment_file, 'r')

    for i, peak in peaks_df.iterrows():
        try:
            fragments_in_peak = tabixfile.fetch(peak['chrom'], peak['chromStart'], peak['chromEnd'])
            for fragment in fragments_in_peak:
                barcode = fragment.split('\t')[3]
                if barcode in barcode_map:
                    matrix[i, barcode_map[barcode]] += 1
        except ValueError:
            print(f"Warning: Chromosome '{peak['chrom']}' not found in fragment file. Skipping.")
            continue
        
        if (i + 1) % 20000 == 0:
            print(f"  Processed {i + 1}/{len(peaks_df)} peaks...")
    
    print("-> Matrix construction complete.")

    # Save the output
    print("\nStep 4: Saving the output files...")
    csr = matrix.tocsr()
    mmwrite(output_matrix_file, csr)
    peaks_df['name'] = peaks_df['peak_id']
    peaks_df['type'] = 'Peak'
    peaks_df[['peak_id', 'name', 'type']].to_csv(output_peaks_file, sep='\t', header=False, index=False)
    with open(output_barcodes_file, 'w') as f:
        f.write("\n".join(barcodes))

    print("\n✅ Done!")
    print(f"Output files are saved in '{output_dir}'")


def setup_multimodal_data(
    data_path: str,
    rna_file: str = "counts_unfiltered/adata.h5ad",
    spatial_file: str = "spatial/tissue_positions_list.csv",
    atac_path: str = "atac"
) -> mu.MuData:
    """
    Sets up multi-modal data combining RNA, ATAC, and spatial information.
    
    Args:
        data_path: Base data directory path.
        rna_file: Path to RNA data file relative to data_path.
        spatial_file: Path to spatial coordinates file relative to data_path.
        atac_path: Path to ATAC data directory relative to data_path.
        
    Returns:
        Processed MuData object.
    """
    # Load RNA data
    rna_path = os.path.join(data_path, rna_file)
    
    adata_rna = dyn.read_h5ad(rna_path)
    if 'mature' in adata_rna.layers:
        adata_rna.layers['spliced'] = adata_rna.layers.pop('mature')
    if 'nascent' in adata_rna.layers:
        adata_rna.layers['unspliced'] = adata_rna.layers.pop('nascent')
    
    # Preprocess RNA data
    pp = dyn.pp.Preprocessor()
    pp.preprocess_adata(adata_rna, recipe='monocle')
    
    # Load ATAC data
    adata_atac = mu.atac.read_10x_mtx(os.path.join(data_path, atac_path))
    
    # Create MuData object
    adata = mu.MuData({'atac': adata_atac, 'rna': adata_rna})
    
    # Load spatial coordinates
    spatial_path = os.path.join(data_path, spatial_file)
    if os.path.exists(spatial_path):
        df_spatial = pd.read_csv(spatial_path, header=None)
        df_spatial.set_index(0, inplace=True)
        df_spatial.drop(columns=[1], inplace=True)
        df_spatial.columns = ['x_pixel', 'y_pixel', 'x_position', 'y_position']
        adata.obs = df_spatial.loc[adata.obs_names]
    
    return adata


def create_graph_data(
    adata: mu.MuData,
    n_neighbors_spatial: int = 8,
    n_neighbors_expression: int = 30,
    use_unspliced: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates graph edge indices for spatial and expression data.
    
    Args:
        adata: MuData object containing the data.
        n_neighbors_spatial: Number of spatial neighbors.
        n_neighbors_expression: Number of expression neighbors.
        use_unspliced: Whether to use unspliced data for expression graph.
        
    Returns:
        Tuple of (spatial_edge_index, expression_edge_index, distance_matrix).
    """
    # Create spatial graph
    spatial_coords = adata.obs[['x_position', 'y_position']].values
    spatial_graph = kneighbors_graph(
        spatial_coords, n_neighbors_spatial, 
        mode='connectivity', include_self=False
    )
    spatial_edge_coo = spatial_graph.tocoo()
    spatial_edge_index = torch.tensor(
        np.vstack((spatial_edge_coo.row, spatial_edge_coo.col)), 
        dtype=torch.long
    )
    
    # Create expression graph
    if use_unspliced:
        expr_data = np.hstack([
            adata['rna'].layers['M_u'].toarray(),
            adata['rna'].layers['M_s'].toarray()
        ])
    else:
        expr_data = adata['rna'].layers['M_s'].toarray()
    
    expr_graph = kneighbors_graph(
        expr_data, n_neighbors_expression,
        mode='connectivity', include_self=False
    )
    expr_edge_coo = expr_graph.tocoo()
    expression_edge_index = torch.tensor(
        np.vstack((expr_edge_coo.row, expr_edge_coo.col)),
        dtype=torch.long
    )
    
    # Create distance matrix for expression data
    expr_tensor = torch.tensor(expr_data, dtype=torch.float32)
    dist_matrix = torch.cdist(expr_tensor, expr_tensor, p=2)
    
    return spatial_edge_index, expression_edge_index, dist_matrix