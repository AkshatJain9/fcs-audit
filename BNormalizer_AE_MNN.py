import numpy as np
import torch
import torch.nn as nn
import hnswlib


def find_mutual_nearest_neighbors_approx(X, Y, k=20, ef=200):
    # Initialize HNSW index for dataset X
    dim = X.shape[1]
    num_elements_X = X.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements_X, ef_construction=200, M=16)
    p.add_items(X)
    p.set_ef(ef)

    # Query Y in the index built on X
    indices_x, _ = p.knn_query(Y, k=k)

    # Initialize HNSW index for dataset Y
    num_elements_Y = Y.shape[0]
    q = hnswlib.Index(space='l2', dim=dim)
    q.init_index(max_elements=num_elements_Y, ef_construction=200, M=16)
    q.add_items(Y)
    q.set_ef(ef)

    # Query X in the index built on Y
    indices_y, _ = q.knn_query(X, k=k)

    # Find mutual nearest neighbors
    mnn_pairs = []
    for i in range(indices_x.shape[0]):
        for x_idx in indices_x[i]:
            if i in indices_y[x_idx]:
                mnn_pairs.append((int(x_idx), int(i)))
                break

    return np.array(mnn_pairs, dtype=np.int64)

def bnormalizer_ae_mnn(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict, k: int = 20):
    device = next(bnormalizer.parameters()).device
    ref_batch_encoded = bnormalizer.encode(ref_batch).detach().cpu().numpy()
    normalised_batches = {}
    
    for key, target_batch in target_batches.items():
        target_batch_encoded = bnormalizer.encode(target_batch).detach().cpu().numpy()
        
        # Find Mutual Nearest Neighbors using approximate methods
        mnn_pairs = find_mutual_nearest_neighbors_approx(ref_batch_encoded, target_batch_encoded, k)
        
        # Ensure mnn_pairs is not empty
        if len(mnn_pairs) == 0:
            print(f"Warning: No mutual nearest neighbors found for batch {key}")
            continue
        
        # Calculate the mean shift between MNN pairs
        ref_mnn = ref_batch_encoded[mnn_pairs[:, 0]]
        target_mnn = target_batch_encoded[mnn_pairs[:, 1]]
        mean_shift = np.mean(ref_mnn - target_mnn, axis=0)
        
        # Apply the mean shift to the entire target batch
        aligned_batch_encoded = target_batch_encoded + mean_shift
        
        # Decode the aligned batch in batches to save memory
        batch_size = 10000  # Adjust based on your GPU memory
        aligned_batch_tensor = torch.from_numpy(aligned_batch_encoded).float().to(device)
        normalised_batch = []
        
        for i in range(0, aligned_batch_tensor.shape[0], batch_size):
            batch = aligned_batch_tensor[i:i+batch_size]
            normalised_batch.append(bnormalizer.decode(batch).detach().cpu().numpy())
        
        normalised_batches[key] = np.concatenate(normalised_batch)
    
    return normalised_batches