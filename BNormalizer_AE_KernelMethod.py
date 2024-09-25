import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

def bnormalizer_ae_kernel(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    device = next(bnormalizer.parameters()).device
    
    # Encode reference batch
    ref_batch = ref_batch.to(device)
    ref_batch_encoded = bnormalizer.encode(ref_batch).detach().cpu().numpy()
    
    # Encode target batches
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    normalised_batches = {}
    
    for key, target_batch_encoded in target_batches_encoded.items():
        normalized_encoded = np.zeros_like(target_batch_encoded)
        
        for feature in range(target_batch_encoded.shape[1]):
            # Estimate reference and target distributions using KDE
            ref_kde = gaussian_kde(ref_batch_encoded[:, feature])
            target_kde = gaussian_kde(target_batch_encoded[:, feature])
            
            # Create a grid of points
            x_min = min(ref_batch_encoded[:, feature].min(), target_batch_encoded[:, feature].min())
            x_max = max(ref_batch_encoded[:, feature].max(), target_batch_encoded[:, feature].max())
            x_grid = np.linspace(x_min, x_max, 1000)
            
            # Estimate CDFs
            ref_cdf = np.cumsum(ref_kde(x_grid)) / np.sum(ref_kde(x_grid))
            target_cdf = np.cumsum(target_kde(x_grid)) / np.sum(target_kde(x_grid))
            
            # Interpolate to find the mapping
            normalized_encoded[:, feature] = np.interp(target_batch_encoded[:, feature], 
                                                       x_grid, 
                                                       np.interp(ref_cdf, target_cdf, x_grid))
        
        # Decode normalized batch
        normalised_batches[key] = bnormalizer.decode(torch.tensor(normalized_encoded, device=device)).cpu().detach().numpy()
    
    return normalised_batches