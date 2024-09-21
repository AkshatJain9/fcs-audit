import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch

def bnormalizer_ae_kmeans(bnormalizer: nn.Module, ref_batch: torch.Tensor, target_batches: dict, n_clusters: int = 5):
    device = next(bnormalizer.parameters()).device
    
    ref_batch_encoded = bnormalizer.encode(ref_batch.to(device)).detach().cpu().numpy()
    target_batches_encoded = {key: bnormalizer.encode(target_batch.to(device)).detach().cpu().numpy()
                              for key, target_batch in target_batches.items()}

    # Perform K-means clustering on reference batch
    ref_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    ref_clusters = ref_kmeans.fit_predict(ref_batch_encoded)
    ref_centroids = ref_kmeans.cluster_centers_

    normalised_batches = {}
    for key, target_batch_encoded in target_batches_encoded.items():
        # Perform K-means clustering on target batch
        target_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        target_clusters = target_kmeans.fit_predict(target_batch_encoded)
        target_centroids = target_kmeans.cluster_centers_

        # Calculate distances between centroids
        distances = np.linalg.norm(ref_centroids[:, np.newaxis] - target_centroids, axis=2)

        # Use Hungarian algorithm to find optimal matching
        ref_indices, target_indices = linear_sum_assignment(distances)

        # Create a mapping from target cluster to reference cluster
        cluster_mapping = dict(zip(target_indices, ref_indices))

        # Apply the transformation to each point in the target batch
        transformed_batch = np.zeros_like(target_batch_encoded)
        for i in range(n_clusters):
            mask = (target_clusters == target_indices[i])
            shift = ref_centroids[cluster_mapping[target_indices[i]]] - target_centroids[target_indices[i]]
            transformed_batch[mask] = target_batch_encoded[mask] + shift

        # Decode the transformed batch
        normalised_batches[key] = bnormalizer.decode(torch.from_numpy(transformed_batch).float().to(device)).detach().cpu().numpy()

    return normalised_batches