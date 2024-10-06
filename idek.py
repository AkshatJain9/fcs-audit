import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Define the function
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
    bin_centers = torch.linspace(min_val + bin_centre_offset, max_val - bin_centre_offset, num_bins).to(feature_values.device)

    # Calculate the bin width
    bin_width = (max_val - min_val) / num_bins
    
    # Compute bin probabilities
    feature_values_expanded = feature_values.unsqueeze(1)
    bin_centers_expanded = bin_centers.unsqueeze(0)
    
    # Softmax-like bin assignment
    distances = torch.abs(feature_values_expanded - bin_centers_expanded)
    bin_probs = torch.exp(-distances / (2 * bin_width))
    
    # Normalize bin probabilities
    bin_probs_sum = bin_probs.sum(dim=0)
    histogram = bin_probs_sum / bin_probs_sum.sum()
    
    return histogram, bin_centers

# Generate thousands of points between -0.5 and 0.5
num_points = 500000

# Define the means and standard deviations for the three Gaussian distributions
means = [-0.3, 0.0, 0.3]
std_devs = [0.05, 0.1, 0.07]

# Generate samples for each Gaussian component
samples1 = torch.normal(means[0], std_devs[0], size=(num_points // 3,))
samples2 = torch.normal(means[1], std_devs[1], size=(num_points // 3,))
samples3 = torch.normal(means[2], std_devs[2], size=(num_points - 2 * (num_points // 3),))

# Combine the samples into one tensor
feature_values = torch.cat([samples1, samples2, samples3])
feature_values_hist, bin_centers = generate_hist(feature_values, num_bins=200, min_val=-0.5, max_val=0.5)

# Sample data based on histogram distribution
hist_cdf = torch.cumsum(feature_values_hist, dim=0)
rand_samples = torch.rand(num_points)  # Uniform random samples between 0 and 1
bin_indices = torch.searchsorted(hist_cdf, rand_samples)

# Map bin indices to bin centers to generate the final samples
generated_samples = bin_centers[bin_indices].numpy()

# Plot the histogram of generated samples
plt.hist(feature_values.numpy(), bins=200, density=True, alpha=0.7, label="True Distribution")
plt.hist(generated_samples, bins=200, density=True, alpha=0.7, label="Approximated Distribution")
plt.legend()    
plt.title("Generated Samples Following Feature Values Histogram")
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.show()
