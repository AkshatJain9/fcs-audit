import numpy as np
import torch.nn as nn
import torch

def bnormalizer_ae_identity(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict):
    normalised_batches = dict()
    for key, target_batch in target_batches.items():
        normalised_batches[key] = bnormalizer(target_batch)[0].detach().cpu().numpy()

    return normalised_batches

