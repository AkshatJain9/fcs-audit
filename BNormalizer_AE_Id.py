from BNormalizer_AE import BNorm_AE
import numpy as np


def bnormalizer_ae_identity(bnormalizer: BNorm_AE, ref_batch: np.ndarray, target_batches: dict):
    normalised_batches = dict()
    for key, target_batch in target_batches.items():
        normalised_batches[key] = bnormalizer(target_batch)

    return normalised_batches

