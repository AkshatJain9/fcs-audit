from BNormalizer_AE import BNorm_AE
import numpy as np
from scipy import stats

def bnormalizer_ae_combat(bnormalizer: BNorm_AE, ref_batch: np.ndarray, target_batches: dict):
    # Encode all batches
    ref_batch_encoded = bnormalizer.encode(ref_batch)
    target_batches_encoded = {key: bnormalizer.encode(batch) for key, batch in target_batches.items()}
    
    # Combine all encoded batches
    all_batches = [ref_batch_encoded] + list(target_batches_encoded.values())
    batch_labels = ['ref'] + list(target_batches_encoded.keys())
    
    # Estimate overall gene means and variances
    overall_mean = np.mean(np.vstack(all_batches), axis=0)
    overall_var = np.var(np.vstack(all_batches), axis=0)
    
    # Estimate batch effects
    batch_means = {label: np.mean(batch, axis=0) for label, batch in zip(batch_labels, all_batches)}
    batch_vars = {label: np.var(batch, axis=0) for label, batch in zip(batch_labels, all_batches)}
    
    # Estimate empirical priors
    gamma_hat = np.mean(list(batch_means.values()), axis=0)
    tau_squared_hat = np.var(list(batch_means.values()), axis=0)
    
    a_prior = np.mean(overall_var / np.array(list(batch_vars.values())), axis=0) ** 2
    b_prior = np.mean(overall_var / np.array(list(batch_vars.values())), axis=0)
    
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
    normalised_batches = {key: bnormalizer.decode(batch) for key, batch in adjusted_batches.items()}
    
    return normalised_batches