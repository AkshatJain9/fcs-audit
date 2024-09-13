from BNormalizer_AE import BNorm_AE
import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_mutual_nearest_neighbors(X, Y, k=20):
    nn_x = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_y = NearestNeighbors(n_neighbors=k, metric='euclidean')
    
    nn_x.fit(X)
    nn_y.fit(Y)
    
    distances_x, indices_x = nn_x.kneighbors(Y)
    distances_y, indices_y = nn_y.kneighbors(X)
    
    mnn_pairs = []
    for i in range(Y.shape[0]):
        for j, x_idx in enumerate(indices_x[i]):
            if i in indices_y[x_idx]:
                mnn_pairs.append((x_idx, i))
                break
    
    return np.array(mnn_pairs)

def bnormalizer_ae_mnn(bnormalizer: BNorm_AE, ref_batch: np.ndarray, target_batches: dict, k: int = 20):
    ref_batch_encoded = bnormalizer.encode(ref_batch)
    target_batches_encoded = {key: bnormalizer.encode(target_batch) for key, target_batch in target_batches.items()}

    normalised_batches = {}
    for key, target_batch_encoded in target_batches_encoded.items():
        # Find Mutual Nearest Neighbors
        mnn_pairs = find_mutual_nearest_neighbors(ref_batch_encoded, target_batch_encoded, k)
        
        # Calculate the mean shift between MNN pairs
        ref_mnn = ref_batch_encoded[mnn_pairs[:, 0]]
        target_mnn = target_batch_encoded[mnn_pairs[:, 1]]
        mean_shift = np.mean(ref_mnn - target_mnn, axis=0)
        
        # Apply the mean shift to the entire target batch
        aligned_batch_encoded = target_batch_encoded + mean_shift
        
        # Decode the aligned batch
        normalised_batches[key] = bnormalizer.decode(aligned_batch_encoded)

    return normalised_batches