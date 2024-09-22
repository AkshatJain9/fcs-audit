import numpy as np
from scipy import stats
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def bnormalizer_ae_combat(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    device = next(bnormalizer.parameters()).device
    
    # Encode all batches
    ref_batch_encoded = bnormalizer.encode(ref_batch.to(device)).detach().cpu().numpy()
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    ref_batch_mean = np.mean(ref_batch_encoded, axis=0)
    ref_batch_var = np.var(ref_batch_encoded, axis=0)

    # Normalise target batches
    target_batches_norm = {}
    for key, target_batch_encoded in target_batches_encoded.items():
        target_batch_normalised = (target_batch_encoded - ref_batch_mean) / np.sqrt(ref_batch_var)
        target_batches_norm[key] = target_batch_normalised

    batch_gamma_hats = {}
    for key, target_batch_norm in target_batches_norm.items():
        batch_gamma_hats[key] = np.mean(target_batch_norm, axis=0)

    batch_gamma_hats_mean = {}
    for key, target_batch_norm in target_batches_norm.items():
        batch_gamma_hats_mean[key] = np.mean(batch_gamma_hats[key])
    
    num_elements_batch = {}
    for key, target_batch_norm in target_batches_norm.items():
        num_elements_batch[key] = target_batch_norm.shape[0]

    tau_squared = {}
    for key, target_batch_norm in target_batches_norm.items():
        batch_gamma_hat = batch_gamma_hats[key]
        batch_gamma_hat_mean = batch_gamma_hats_mean[key]
        tau_squared[key] = np.sum((batch_gamma_hat - batch_gamma_hat_mean) ** 2) / (num_elements_batch[key] - 1)

    # Step 5: Compute mean and variance of variances for each batch (method of moments)
    mu_batch = {}  # mean of variances
    sigma_squared_batch = {}  # variance of variances

    for key, target_batch_norm in target_batches_norm.items():
        # Compute gene-wise variances (s_ig^2) across samples in each batch
        gene_wise_variances = np.var(target_batch_norm, axis=0)
        
        # Compute the mean (mu) of variances and variance (sigma^2) of those variances
        mu_batch[key] = np.mean(gene_wise_variances)
        sigma_squared_batch[key] = np.var(gene_wise_variances)

    # Step 6: Compute theta and lambda for each batch using the method of moments
    lambda_batch = {}
    theta_batch = {}

    for key in target_batches_norm.keys():
        mu = mu_batch[key]
        sigma_squared = sigma_squared_batch[key]

        # Compute lambda using the formula: lambda = 2 + (mu^2 / sigma^2)
        lambda_batch[key] = 2 + (mu ** 2 / sigma_squared)

        # Compute theta using the formula: theta = mu * (lambda - 1)
        theta_batch[key] = mu * (lambda_batch[key] - 1)

    gamma_init = {}  # Initialize gamma values (additive batch effect)
    delta_init = {}  # Initialize delta values (multiplicative batch effect)

    for key, target_batch_norm in target_batches_norm.items():
        # Compute the gene-wise means (gamma initialization)
        gamma_init[key] = np.mean(target_batch_norm, axis=0)

        # Compute the gene-wise variances (delta initialization)
        # If you prefer simpler initialization, you can set this to 1.
        delta_init[key] = np.ones_like(gamma_init[key]) # or just np.ones_like(gamma_init[key])

    # Step 4: Run the Empirical Bayes Algorithm iteratively

    gamma_final = {}
    delta_final = {}

    num_g = len(ref_batch_mean)

    for key in target_batches_norm.keys():
        # Initialize additive and multiplicative effects
        gamma_i = gamma_init[key]
        delta_i = delta_init[key]
        
        n_i = num_elements_batch[key]
        tau_i_squared = tau_squared[key]
        lambda_i = lambda_batch[key]
        theta_i = theta_batch[key]
        batch_gamma_hat = batch_gamma_hats[key]
        batch_gamma_hat_mean = batch_gamma_hats_mean[key]

        converged = False
        iter_count = 0

        while not converged and iter_count < 10000:
            iter_count += 1
            prev_gamma_i = gamma_i.copy()
            prev_delta_i = delta_i.copy()

            # Update gamma (additive effect)
            gamma_i = []
            for g in range(num_g):
                numerator = n_i * (tau_i_squared * batch_gamma_hat[g] + delta_i[g] * batch_gamma_hat_mean)
                denominator = n_i * tau_i_squared + delta_i[g]
                gamma_i.append(numerator / denominator)

            gamma_i = np.array(gamma_i)

            delta_i = []
            for g in range(num_g):
                numerator = theta_i + 0.5 * np.sum((target_batches_norm[key][:, g] - gamma_i[g]) ** 2)
                denominator = (n_i / 2) + lambda_i - 1
                delta_i.append(numerator / denominator)

            delta_i = np.array(delta_i)

            # Check for convergence (tolerance check)
            gamma_diff = np.max(np.abs(gamma_i - prev_gamma_i))
            delta_diff = np.max(np.abs(delta_i - prev_delta_i))
            
            if gamma_diff < 1e-6 and delta_diff < 1e-6:
                converged = True

        gamma_final[key] = gamma_i
        delta_final[key] = delta_i

    # Step 5: Adjust the data for batch effects
    adjusted_batches = {}
    for key, target_batch_norm in target_batches_norm.items():
        gamma_i = gamma_final[key]
        delta_i = delta_final[key]
        
        # Adjust the data for batch effects
        adjusted_batch = (target_batch_norm - gamma_i) / (np.sqrt(delta_i))
        adjusted_batch = (adjusted_batch * np.sqrt(ref_batch_var)) + ref_batch_mean
        adjusted_batches[key] = adjusted_batch

    for key, adjusted_batch in adjusted_batches.items():
        # Decode the adjusted batch
        plot_all_histograms(ref_batch_encoded, adjusted_batch)
        assert False

    normalised_batches = {}
    for key, adjusted_batch in adjusted_batches.items():
        # Decode the adjusted batch
        normalised_batches[key] = bnormalizer.decode(torch.from_numpy(adjusted_batch).float().to(device)).detach().cpu().numpy()


    
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