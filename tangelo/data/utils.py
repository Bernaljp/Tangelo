"""Utility functions for data processing."""

from typing import Tuple, Union
import torch
import numpy as np


def get_cdf(data: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the empirical Cumulative Distribution Function (CDF) for each feature.
    This is used for pre-training the sigmoid function.

    Args:
        data: Input data of shape (n_samples, n_features).

    Returns:
        Tuple containing:
            - x_all: Sorted feature values, shape (n_features, n_samples).
            - y_all: Corresponding CDF values, shape (n_features, n_samples).
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    device = data.device
    n_samples = data.shape[0]
    x_all, y_all = [], []

    # Transpose to iterate over features
    for x_feature in data.T:
        x_valid = x_feature[x_feature > 0]
        if len(x_valid) == 0:
            # Handle cases where all values are zero
            x_all.append(torch.zeros_like(x_feature, device=device))
            y_all.append(torch.zeros_like(x_feature, device=device))
            continue
            
        x_sorted, _ = torch.sort(x_feature)
        x_valid_sorted, _ = torch.sort(x_valid)
        
        # Create CDF for non-zero values
        y_valid = torch.linspace(0, 1, len(x_valid_sorted), device=device)
        
        # The CDF is 0 for all zero values
        num_zeros = n_samples - len(x_valid_sorted)
        y_full_cdf = torch.cat([torch.zeros(num_zeros, device=device), y_valid])
        
        x_all.append(x_sorted)
        y_all.append(y_full_cdf)
        
    return torch.stack(x_all), torch.stack(y_all)