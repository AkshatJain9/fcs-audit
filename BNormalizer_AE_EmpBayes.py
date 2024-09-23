import numpy as np
from scipy import stats
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def bnormalizer_ae_combat(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    device = next(bnormalizer.parameters()).device
    
    # Encode all batches
    ref_batch_encoded = bnormalizer.encode(ref_batch.to(device)).detach().cpu().numpy()
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    ref_batch_gmms = []
    for i in range(ref_batch_encoded.shape[1]):
        gmm = GaussianMixture(n_components=5)
        gmm.fit(ref_batch_encoded[:, i].reshape(-1, 1))
        ref_batch_gmms.append(gmm)
    
    
    for key, target_batch_encoded in target_batches_encoded.items():
        for i in range(target_batch_encoded.shape[1]):
            # Fit the Gaussian Mixture Model
            gmm = GaussianMixture(n_components=5)
            gmm.fit(target_batch_encoded[:, i].reshape(-1, 1))

            # Transform the target distribution to match the reference distribution, use shrinkage as is done in ComBat
            
            

            
    for key, target_batch_encoded in target_batches_encoded.items():
        plot_all_histograms(ref_batch_encoded, target_batch_encoded)

    normalised_batches = {}
    for key, target_batch_encoded in target_batches_encoded.items():
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