import numpy as np
from scipy import stats
import torch.nn as nn
import torch

def bnormalizer_ae_combat(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    device = next(bnormalizer.parameters()).device
    
    # Encode all batches
    ref_batch_encoded = bnormalizer.encode(ref_batch.to(device)).detach().cpu().numpy()
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    # Combine all encoded batches
    all_batches = [ref_batch_encoded] + list(target_batches_encoded.values())
    batch_labels = ['ref'] + list(target_batches_encoded.keys())
    
    # Estimate overall gene means and variances
    all_batches_stacked = np.vstack(all_batches)
    overall_mean = np.mean(all_batches_stacked, axis=0)
    overall_var = np.var(all_batches_stacked, axis=0)
    
    # Estimate batch effects
    batch_means = {label: np.mean(batch, axis=0) for label, batch in zip(batch_labels, all_batches)}
    batch_vars = {label: np.var(batch, axis=0) for label, batch in zip(batch_labels, all_batches)}
    
    # Estimate empirical priors
    batch_means_array = np.array(list(batch_means.values()))
    batch_vars_array = np.array(list(batch_vars.values()))
    gamma_hat = np.mean(batch_means_array, axis=0)
    tau_squared_hat = np.var(batch_means_array, axis=0)
    a_prior = np.mean(overall_var / batch_vars_array, axis=0) ** 2
    b_prior = np.mean(overall_var / batch_vars_array, axis=0)
    
    # Estimate posterior parameters
    gamma_star = {}
    delta_star = {}
    for label, batch in zip(batch_labels, all_batches):
        n_samples = batch.shape[0]
        gamma_star[label] = (tau_squared_hat * batch_means[label] + batch_vars[label] * gamma_hat) / (tau_squared_hat + batch_vars[label] / n_samples)
        delta_star[label] = (n_samples * batch_vars[label] + b_prior * (a_prior + n_samples / 2)) / (a_prior + n_samples / 2 - 1)
    
    # Adjust target batches to align with reference batch
    adjusted_batches = {}
    for label, batch in target_batches_encoded.items():
        # Standardize using overall mean and batch-specific variance
        standardized = (batch - overall_mean) / np.sqrt(batch_vars[label])
        # Adjust using reference batch parameters
        adjusted = standardized * np.sqrt(delta_star['ref']) + gamma_star['ref']
        adjusted_batches[label] = adjusted
    
    # Decode adjusted batches
    normalised_batches = {key: bnormalizer.decode(torch.from_numpy(batch).float().to(device)).detach().cpu().numpy()
                          for key, batch in adjusted_batches.items()}
    
    return normalised_batches