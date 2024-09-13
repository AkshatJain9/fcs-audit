from BNormalizer_AE import BNorm_AE
import numpy as np


def bnormalizer_ae_linear(bnormalizer: BNorm_AE, ref_batch: np.ndarray, target_batches: dict):
    ref_batch_encoded = bnormalizer.encode(ref_batch)
    target_batches_encoded = dict()
    for key, target_batch in target_batches.items():
        target_batches_encoded[key] = bnormalizer.encode(target_batch)


    ref_batch_mean = np.mean(ref_batch_encoded, axis=0)
    ref_batch_std = np.std(ref_batch_encoded, axis=0)

    normalised_batches = dict()
    for key, target_batch in target_batches.items():
        target_batch_encoded = target_batches_encoded[key]
        target_batch_mean = np.mean(target_batch_encoded, axis=0)
        target_batch_std = np.std(target_batch_encoded, axis=0)

        target_batch_encoded = (target_batch_encoded - target_batch_mean) / target_batch_std
        target_batch_encoded = target_batch_encoded * ref_batch_std + ref_batch_mean

        normalised_batches[key] = bnormalizer.decode(target_batch_encoded)

    return normalised_batches

