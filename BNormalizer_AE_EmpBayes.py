import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

def subsample_data(data, percentage):
    """
    Subsample the data by a given percentage.
    """
    sample_size = int(len(data) * percentage)
    indices = np.random.choice(np.arange(len(data)), sample_size, replace=False)
    return np.copy(data[indices])

def bnormalizer_ae_combat(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    n_components = 5

    device = next(bnormalizer.parameters()).device
    
    # Encode all batches
    ref_batch_encoded = bnormalizer.encode(ref_batch.to(device)).detach().cpu().numpy()
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    ref_batch_encoded_subsampled = subsample_data(ref_batch_encoded, 0.01)
    ref_batch_gmms = []
    for i in range(ref_batch_encoded.shape[1]):
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm.fit(ref_batch_encoded_subsampled[:, i].reshape(-1, 1))
        
        # Sort the GMM components by means
        sorted_indices = np.argsort(gmm.means_.flatten())
        gmm.means_ = gmm.means_[sorted_indices]
        gmm.covariances_ = gmm.covariances_[sorted_indices]
        
    # Visualize the GMMs fitted on the reference batch
    visualize_gmms_on_ref(ref_batch_encoded_subsampled, ref_batch_gmms)
    assert False
    
    adjusted_target_batches = {}
    for key, target_batch_encoded in target_batches_encoded.items():
        target_batch_shifted = target_batch_encoded
        target_batch_encoded_subsampled = subsample_data(target_batch_encoded, 0.001)
        for i in range(target_batch_encoded.shape[1]):  # For each feature
            
            # Fit a GMM to the current feature in the target batch
            gmm_target = GaussianMixture(n_components=n_components, covariance_type='diag')
            gmm_target.fit(target_batch_encoded_subsampled[:, i].reshape(-1, 1))
            
            # Sort the GMM components by means
            sorted_indices = np.argsort(gmm_target.means_.flatten())
            gmm_target.means_ = gmm_target.means_[sorted_indices]
            gmm_target.covariances_ = gmm_target.covariances_[sorted_indices]

            # Store changes in target_batch_shifted
            # Get the reference GMM for this feature
            gmm_ref = ref_batch_gmms[i]
            
            # Compute the shift for each Gaussian component pair
            shifts = gmm_ref.means_.flatten() - gmm_target.means_.flatten()
            
            # Compute responsibilities for each data point in the target batch
            responsibilities = gmm_target.predict_proba(target_batch_encoded[:, i].reshape(-1, 1))
            
            # Apply the shift weighted by the responsibilities
            for k in range(n_components):  # For each Gaussian component
                # Compute shrinkage as the ratio of the target component's covariance to the reference component's covariance
                target_batch_shifted[:, i] += responsibilities[:, k] * shifts[k]

        
        # Store the shifted (adjusted) target batch
        adjusted_target_batches[key] = target_batch_shifted

    # Plot histograms for comparison (optional step for debugging or visualization)
    for key, target_batch_encoded in adjusted_target_batches.items():
        plot_all_histograms(ref_batch_encoded, target_batch_encoded)

    normalised_batches = {}
    for key, target_batch_encoded in adjusted_target_batches.items():
        normalised_batches[key] = bnormalizer.decode(torch.Tensor(target_batch_encoded).to(device)).detach().cpu().numpy()

    return normalised_batches

#################################
def smooth_gaussian(data, sigma=2):
    smoothed_data = np.copy(data)
    for col in range(smoothed_data.shape[1]):
        smoothed_data[:, col] = gaussian_filter1d(smoothed_data[:, col], sigma=sigma)
    return smoothed_data

def smooth_savitzky_golay(data, window_length=5, polyorder=2):
    smoothed_data = np.copy(data)
    for col in range(smoothed_data.shape[1]):
        smoothed_data[:, col] = savgol_filter(smoothed_data[:, col], window_length=window_length, polyorder=polyorder)
    return smoothed_data

def apply_median_filter(data, size=3):
    smoothed_data = np.copy(data)
    for col in range(smoothed_data.shape[1]):
        # Apply median filter along each column
        smoothed_data[:, col] = median_filter(smoothed_data[:, col], size=size)
    return smoothed_data

def visualize_gmms_on_ref(ref_batch_encoded_subsampled, ref_batch_gmms):
    """
    Visualizes the GMMs fitted on the reference batch for each feature (column).
    """
    num_features = ref_batch_encoded_subsampled.shape[1]
    x = np.linspace(np.min(ref_batch_encoded_subsampled), np.max(ref_batch_encoded_subsampled), 1000)

    fig, axs = plt.subplots(num_features, 1, figsize=(8, 2*num_features))
    
    # If there's only one subplot, axs won't be an array, so wrap it in one
    if num_features == 1:
        axs = [axs]
    
    for i, ax in enumerate(axs):
        # Plot histogram of data for this feature
        ax.hist(ref_batch_encoded_subsampled[:, i], bins=100, density=True, alpha=0.6, label='Data')
        
        # Plot the GMM components
        gmm = ref_batch_gmms[i]
        for j in range(gmm.means_.shape[0]):
            mean = gmm.means_[j, 0]
            cov = gmm.covariances_[j, 0]
            weight = gmm.weights_[j]
            # Plot the Gaussian component
            y = weight * (1.0 / np.sqrt(2 * np.pi * cov)) * np.exp(-(x - mean) ** 2 / (2 * cov))
            ax.plot(x, y, label=f'Component {j+1}')
        
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def generate_histogram(panel_np, index, min_val, max_val):
    range = (min_val, max_val)

    hist, _ = np.histogram(panel_np[:, index], bins=200, range=range)
    hist = hist / np.sum(hist)
    return hist

def plot_all_histograms(panel_np_1, panel_np_2):
    num_channels = panel_np_1.shape[1]
    
    # Create a figure with subplots, one for each channel
    fig, axs = plt.subplots(num_channels, 1, figsize=(8, 2*num_channels))
    
    # If there's only one subplot, axs won't be an array, so wrap it in one
    if num_channels == 1:
        axs = [axs]
    
    for i, ax in enumerate(axs):
        min_val = np.min([np.min(panel_np_1[:, i]), np.min(panel_np_2[:, i])])
        max_val = np.max([np.max(panel_np_1[:, i]), np.max(panel_np_2[:, i])])
        hist1 = generate_histogram(panel_np_1, i, min_val, max_val)
        hist2 = generate_histogram(panel_np_2, i, min_val, max_val)
        
        # Plot on the current axis
        ax.plot(hist1, label='Reference')
        ax.plot(hist2, label='Panel 2')
        ax.legend()
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()