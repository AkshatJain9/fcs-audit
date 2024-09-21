import numpy as np
from scipy.interpolate import PchipInterpolator
import torch.nn as nn
import torch


def bnormalizer_ae_splinefn(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    device = next(bnormalizer.parameters()).device
    
    # Encode reference batch
    ref_batch = ref_batch.to(device)
    ref_batch_encoded = bnormalizer.encode(ref_batch).detach().cpu().numpy()
    
    # Encode target batches
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach().cpu().numpy() 
                              for key, batch in target_batches.items()}
    
    # Create spline functions
    spline_fns = {key: Spline_Transform(ref_batch_encoded, encoded_batch)
                  for key, encoded_batch in target_batches_encoded.items()}
    
    # Apply transformations and decode
    normalised_batches = {}
    for key, target_batch in target_batches.items():
        transformed = spline_fns[key].transform(target_batches_encoded[key])
        normalised_batches[key] = bnormalizer.decode(torch.from_numpy(transformed).float().to(device)).cpu().detach().numpy()
    
    return normalised_batches

class Spline_Transform():
    def __init__(self, ref_batch: np.ndarray, target_batch: np.ndarray):
        self.interpolators = []
        for i in range(ref_batch.shape[1]):
            ref_col = ref_batch[:, i]
            target_col = target_batch[:, i]
            target_col += 1e-6
            ref_col_quantiles_y = np.percentile(ref_col, np.linspace(0, 100, 101))
            target_col_quantiles_x = np.percentile(target_col, np.linspace(0, 100, 101))
            self.interpolators.append(PchipInterpolator(target_col_quantiles_x, ref_col_quantiles_y))
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        transformed_data = np.zeros(data.shape)
        for i in range(data.shape[1]):
            transformed_data[:, i] = self.interpolators[i](data[:, i])
        return transformed_data


