import numpy as np
import torch

def generate_hist(feature_values, num_bins, min_val, max_val):
    """
    Generate a histogram from a tensor of values, in a differentiable manner.
    
    Parameters:
    - feature_values: Tensor of feature values (shape: [N]).
    - num_bins: Number of bins in the histogram.
    - min_val: Minimum value in the range of the histogram.
    - max_val: Maximum value in the range of the histogram.

    Returns:
    - Tensor representing the histogram (shape: [num_bins]).
    """
    bin_centre_offset = (max_val - min_val) / (2 *num_bins)
    # Define bin centers
    bin_centers = torch.linspace(min_val + bin_centre_offset, max_val - bin_centre_offset, num_bins)

    # Calculate the bin width
    bin_width = (max_val - min_val) / num_bins
    
    # Compute bin probabilities
    feature_values_expanded = feature_values.unsqueeze(1)

    bin_centers_expanded = bin_centers.unsqueeze(0)
    
    # Softmax-like bin assignment
    distances = torch.abs(feature_values_expanded - bin_centers_expanded)
    bin_probs = torch.exp(-distances / (bin_width))
    
    # Normalize bin probabilities
    bin_probs_sum = bin_probs.sum(dim=0)
    histogram = bin_probs_sum / bin_probs_sum.sum()
    
    return histogram

feature_values = torch.tensor([-0.75, -0.75, 0, 0.75], dtype=torch.float32)
num_bins = 4
min_val = -1.0
max_val = 1.0

histogram = generate_hist(feature_values, num_bins, min_val, max_val)
print(histogram)