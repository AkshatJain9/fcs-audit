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
    gamma_hat_map = {}
    sigma_sq_map = {}
    for key, target_batch_encoded in target_batches_encoded.items():
        sigma_sq_map[key] = np.var(target_batch_encoded, axis=0)
        target_batch_normalised = (target_batch_encoded - ref_batch_mean) / np.sqrt(ref_batch_var)
        target_batches_norm[key] = target_batch_normalised
        gamma_hat_map[key] = np.mean(target_batch_normalised, axis=0)

    ref_batch_encoded_norm = (ref_batch_encoded - ref_batch_mean) / np.sqrt(ref_batch_var)
    for key, target_batch_norm in target_batches_norm.items():
        # Decode the normalised batch
        plot_all_histograms(ref_batch_encoded_norm, target_batch_norm)
    assert False


    gamma_hat_values = np.array(list(gamma_hat_map.values()))
    gamma_bar = np.mean(gamma_hat_values, axis=0)
    t2 = np.var(gamma_hat_values, axis=0)

    # Step 4: Run the Empirical Bayes Algorithm iteratively
    gamma_final = {}
    delta_final = {}
    
    for key, target_batch_norm in target_batches_norm.items():
        gamma_hats = np.mean(target_batch_norm, axis=0)
        delta_hats = np.var(target_batch_norm, axis=0)
        sigma_squared = sigma_sq_map[key]
        
        a_prior = (2 * np.mean(delta_hats) ** 2) / np.var(delta_hats)
        b_prior = (2 * np.mean(delta_hats) ** 3) / np.var(delta_hats)


        gamma_star = []
        delta_star = []
        for i in range(len(gamma_hats)):
            gamma_hat_i = gamma_hats[i]
            sigma_sq = sigma_squared[i]
            gamma_bar_i = gamma_bar[i]
            t2_i = t2[i]

            numerator = (gamma_hat_i / sigma_sq) + (gamma_bar_i / t2_i)
            denominator = (1 / sigma_sq) + (1 / t2_i)
            gamma_star_i = numerator / denominator
            gamma_star.append(gamma_star_i)

            delta_hat_i = delta_hats[i]
            n_i = len(target_batch_norm)
            delta_star_i = (delta_hat_i + b_prior) / (n_i + a_prior)

            delta_star.append(delta_star_i)

        gamma_star = np.array(gamma_star)
        gamma_final[key] = gamma_star

        delta_star = np.array(delta_star)
        delta_final[key] = delta_star

    # Step 5: Adjust the data for batch effects
    adjusted_batches = {}
    for key, target_batch_norm in target_batches_norm.items():
        gamma_i = gamma_final[key]
        delta_i = 1 + delta_final[key]

        # Adjust the data for batch effects
        adjusted_batch = (target_batch_norm - gamma_i) / np.sqrt(delta_i)
        adjusted_batch = (adjusted_batch * np.sqrt(ref_batch_var)) + ref_batch_mean
        adjusted_batches[key] = adjusted_batch

    # for key, adjusted_batch in adjusted_batches.items():
    #     # Decode the adjusted batch
    #     plot_all_histograms(ref_batch_encoded, adjusted_batch)

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