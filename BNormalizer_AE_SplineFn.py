from BNormalizer_AE import BNorm_AE
import numpy as np
from scipy.interpolate import PchipInterpolator


def bnormalizer_ae_splinefn(bnormalizer: BNorm_AE, ref_batch: np.ndarray, target_batches: dict):
    ref_batch_encoded = bnormalizer.encode(ref_batch)
    target_batches_encoded = dict()
    for key, target_batch in target_batches.items():
        target_batches_encoded[key] = bnormalizer.encode(target_batch)

    spline_fns = dict()
    for key, target_batch_encoded in target_batches_encoded.items():
        spline_fns[key] = Spline_Transform(ref_batch_encoded, target_batch_encoded)

    normalised_batches = dict()
    for key, target_batch in target_batches.items():
        normalised_batches[key] = bnormalizer.decode(spline_fns[key].transform(target_batches_encoded[key]))

    return normalised_batches

class Spline_Transform():
    def __init__(self, ref_batch: np.ndarray, target_batch: np.ndarray):
        self.interpolators = []

        for i in range(0, ref_batch.shape[1]):
            ref_col = ref_batch[:, i]
            target_col = target_batch[:, i]
            target_col += 1e-6

            print(target_col)

            ref_col_quantiles_y = np.percentile(ref_col, np.linspace(0, 100, 101))
            target_col_quantiles_x = np.percentile(target_col, np.linspace(0, 100, 101))
            print(target_col_quantiles_x)
            self.interpolators.append(PchipInterpolator(target_col_quantiles_x, ref_col_quantiles_y))

    def transform(self, data: np.ndarray) -> np.ndarray:
        transformed_data = np.zeros(data.shape)
        for i in range(0, data.shape[1]):
            transformed_data[:, i] = self.interpolators[i](data[:, i])
        return transformed_data


