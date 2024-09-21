import numpy as np
import torch.nn as nn
import torch


def bnormalizer_ae_linear(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    device = next(bnormalizer.parameters()).device
    
    # Encode reference batch
    ref_batch = ref_batch.to(device)
    ref_batch_encoded = bnormalizer.encode(ref_batch).detach()
    
    # Encode target batches
    target_batches_encoded = {key: bnormalizer.encode(batch.to(device)).detach() for key, batch in target_batches.items()}
    
    # Calculate reference batch statistics
    ref_batch_mean = ref_batch_encoded.mean(dim=0)
    ref_batch_std = ref_batch_encoded.std(dim=0)
    
    normalised_batches = {}
    for key, target_batch in target_batches.items():
        target_batch_encoded = target_batches_encoded[key]
        
        # Calculate target batch statistics
        target_batch_mean = target_batch_encoded.mean(dim=0)
        target_batch_std = target_batch_encoded.std(dim=0)
        
        # Normalize target batch
        normalized_encoded = ((target_batch_encoded - target_batch_mean) / target_batch_std) * ref_batch_std + ref_batch_mean
        
        # Decode normalized batch
        normalised_batches[key] = bnormalizer.decode(normalized_encoded).cpu().detach().numpy()
    
    return normalised_batches

