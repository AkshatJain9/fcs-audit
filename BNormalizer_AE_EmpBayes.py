import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
    ref_batch_encoded_subsampled = subsample_data(ref_batch_encoded, 0.001)
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    ref_batch_gmms = []
    for i in range(ref_batch_encoded.shape[1]):
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag')
        gmm.fit(ref_batch_encoded_subsampled[:, i].reshape(-1, 1))
        
        # Sort the GMM components by means
        sorted_indices = np.argsort(gmm.means_.flatten())
        gmm.means_ = gmm.means_[sorted_indices]
        gmm.covariances_ = gmm.covariances_[sorted_indices]
        
        ref_batch_gmms.append(gmm)
    
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