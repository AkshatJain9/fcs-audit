import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import flowkit as fk
import glob
import os
import platform
from geomloss import SamplesLoss
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from itertools import combinations

scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels = ['BUV 395-A', 'BUV737-A', 'Pacific Blue-A', 'FITC-A', 'PerCP-Cy5-5-A', 'PE-A', 'PE-Cy7-A', 'APC-A', 'Alexa Fluor 700-A', 'APC-Cy7-A','BV510-A','BV605-A']
new_channels = ['APC-Alexa 750 / APC-Cy7-A', 'Alexa 405 / Pac Blue-A', 'Qdot 605-A']
fluro_channels += new_channels

all_channels = scatter_channels + fluro_channels

transform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)

if (platform.system() == "Windows"):
    somepath = 'C:\\Users\\aksha\\Documents\\ANU\\COMP4550_(Honours)\\Data\\'
else:
    somepath = '/home/akshat/Documents/Data/'
directories = [d for d in os.listdir(somepath) if os.path.isdir(os.path.join(somepath, d))]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BNorm_AE(nn.Module):
    def __init__(self, ch_count, reduce_dims):
        super(BNorm_AE, self).__init__()

        # Remove scatter channels
        self.down1 = nn.Linear(in_features=ch_count, out_features=(ch_count - 3))
        self.down2 = nn.Linear(in_features=(ch_count - 3), out_features=(ch_count - 6))

        # Real dimensionality reduction of fluoro channels
        self.down3 = nn.Linear(in_features=(ch_count - 6), out_features=(ch_count - 6 - (reduce_dims // 3)))
        self.down4 = nn.Linear(in_features=(ch_count - 6 - (reduce_dims // 3)), out_features=(ch_count - 6 - (2 * reduce_dims // 3)))
        self.down5 = nn.Linear(in_features=(ch_count - 6 - (2 * reduce_dims // 3)), out_features=(ch_count - 6 - reduce_dims))

        # Build up back to fluoro dimensionality

        self.up1 = nn.Linear(in_features=(ch_count - 6 - reduce_dims), out_features=(ch_count - 6 - (2 * reduce_dims // 3)))
        self.up2 = nn.Linear(in_features=(ch_count - 6 - (2 * reduce_dims // 3)), out_features=(ch_count - 6 - (reduce_dims // 3)))
        self.up3 = nn.Linear(in_features=(ch_count - 6 - (reduce_dims // 3)), out_features=(ch_count - 6))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, input_data):
        x = self.down1(input_data)
        x = self.relu(x)
        x = self.down2(x)
        x = self.relu(x)
        x = self.down3(x)
        x = self.relu(x)
        x = self.down4(x)
        x = self.relu(x)
        x = self.down5(x)

        y = self.up1(x)
        y = self.relu(y)
        y = self.up2(y)
        y = self.relu(y)
        y = self.up3(y)
        return y, x
    
    def encode(self, input_data):
        with torch.no_grad():
            x = self.down1(input_data)
            x = self.relu(x)
            x = self.down2(x)
            x = self.relu(x)
            x = self.down3(x)
            x = self.relu(x)
            x = self.down4(x)
            x = self.relu(x)
            x = self.down5(x)
            return x
        
    def decode(self, input_data):
        with torch.no_grad():
            y = self.up1(input_data)
            y = self.relu(y)
            y = self.up2(y)
            y = self.relu(y)
            y = self.up3(y)
            return y
        
class BNorm_AE_Overcomplete(nn.Module):
    def __init__(self, ch_count, increase_dims):
        super(BNorm_AE_Overcomplete, self).__init__()

        self.up1 = nn.Linear(in_features=(ch_count), out_features=(ch_count + (increase_dims // 3)))
        self.up2 = nn.Linear(in_features=(ch_count + (increase_dims // 3)), out_features=(ch_count + (2 * increase_dims // 3)))
        self.up3 = nn.Linear(in_features=(ch_count + (2 * increase_dims // 3)), out_features=(ch_count + increase_dims))

        self.down1 = nn.Linear(in_features=(ch_count + increase_dims), out_features=(ch_count + (2 * increase_dims // 3)))
        self.down2 = nn.Linear(in_features=(ch_count + (2 * increase_dims // 3)), out_features=(ch_count + (increase_dims // 3)))
        self.down3 = nn.Linear(in_features=(ch_count + (increase_dims // 3)), out_features=(ch_count))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, input_data):
        x = self.up1(input_data)
        x = self.relu(x)
        x = self.up2(x)
        x = self.relu(x)
        x = self.up3(x)

        y = self.down1(x)
        y = self.relu(y)
        y = self.down2(y)
        y = self.relu(y)
        y = self.down3(y)
        return y, x
    
    def encode(self, input_data):
        with torch.no_grad():
            x = self.up1(input_data)
            x = self.relu(x)
            x = self.up2(x)
            x = self.relu(x)
            x = self.up3(x)
            return x
        
    def decode(self, input_data):
        with torch.no_grad():
            y = self.down1(input_data)
            y = self.relu(y)
            y = self.down2(y)
            y = self.relu(y)
            y = self.down3(y)
            return y
        

def train_model(model: nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                epoch_count: int, 
                learning_rate: float, 
                p: float, 
                cluster_centres: torch.Tensor, 
                cluster_cov: torch.Tensor) -> np.ndarray:
    print("##### STARTING TRAINING OF MODEL #####")
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Functions
    mse_loss = nn.MSELoss()
    sinkhorn_distance = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    total_losses = []
    mse_losses = []
    tvd_losses = []
    cluster_align_losses = []
    vr_complex_losses = []
    
    for epoch in range(epoch_count):
        total_loss = 0.0
        total_samples = 0
        total_mse_loss = 0.0
        total_tvd_loss = 0.0
        total_cluster_align_loss = 0.0
        total_vr_complex_loss = 0.0
        
        for batch in data_loader:
            x = batch[0]
            x = x.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            pred_ae = output[0]
            latent = output[1]
            
            # Calculate MSE and Sinkhorn loss
            mse_loss_ae = mse_loss(pred_ae, x)
            # tvd_loss_val = tvd_loss(pred_ae, x[:, 6:], sinkhorn_distance)
            # cluster_align_loss = assign_clusters_and_compute_mse(pred_ae, cluster_centres, cluster_cov, batch[1])


            # Randomly sample 250 points for vr_complex_loss
            # batch_size = x.shape[0]
            # num_points = min(250, batch_size)  # Ensure we don't sample more than available
            # indices = torch.randperm(batch_size)[:num_points]

            # # Sample the data
            # x_sampled = x[indices, 6:]
            # latent_sampled = latent[indices]

            # Compute vr_complex_loss with the sampled data
            # vr_complex_loss_val = vr_complex_loss_clusters(x_sampled, latent_sampled, batch[1][indices])
            # vr_complex_loss_val = spread_loss(latent, batch[1])
            # vr_complex_loss_val = spread_loss(latent, batch[1])
            
            # Total loss
            # total_loss_ae = 0.9 * (0.3 * mse_loss_ae + 0.7 * tvd_loss_val) + 0.1 * cluster_align_loss
            # total_loss_ae = 0.8 * total_loss_ae + 0.2 * vr_complex_loss_val
            # total_loss_ae = mse_loss_ae + tvd_loss_val + cluster_align_loss + vr_complex_loss_val
            # total_loss_ae = 0.3 * mse_loss_ae + 0.7 * tvd_loss_val
            # total_loss_ae = mse_loss_ae + vr_complex_loss_val

            # mse_loss_ae = assign_clusters_and_compute_mse(pred_ae, cluster_centres, cluster_cov, batch[1])
            # tvd_loss_val = compute_clus_dist_loss(x[:, 6:], pred_ae, cluster_centres, batch[1], sinkhorn_distance)
            # vr_complex_loss_val = vr_complex_loss_clusters(x[:, 6:], pred_ae, batch[1])

            # total_loss_ae = 0.3 * mse_loss_ae + 0.7 * tvd_loss_val
            total_loss_ae = mse_loss_ae

            total_loss_ae.backward()
            optimizer.step()
            
            # Track losses
            total_loss += total_loss_ae.item()
            total_samples += x.size(0)
            total_mse_loss += mse_loss_ae.item()
            # total_tvd_loss += tvd_loss_val.item()
            # total_cluster_align_loss += cluster_align_loss.item()
            # total_vr_complex_loss += vr_complex_loss_val.item()
        
        # Calculate average losses
        avg_loss = total_loss / total_samples
        avg_mse_loss = total_mse_loss / total_samples
        avg_tvd_loss = total_tvd_loss / total_samples
        avg_cluster_align_loss = total_cluster_align_loss / total_samples
        avg_vr_complex_loss = total_vr_complex_loss / total_samples
        
        # Append losses to lists
        total_losses.append(avg_loss)
        mse_losses.append(avg_mse_loss)
        tvd_losses.append(avg_tvd_loss)
        cluster_align_losses.append(avg_cluster_align_loss)
        vr_complex_losses.append(avg_vr_complex_loss)
        
        print(f'Epoch: {epoch} Loss per unit: {avg_loss}')
        print(f'Epoch: {epoch} MSE Loss per unit: {avg_mse_loss}')
        print(f'Epoch: {epoch} TVD Loss per unit: {avg_tvd_loss}')
        print(f'Epoch: {epoch} Cluster Alignment Loss per unit: {avg_cluster_align_loss}')
        print(f'Epoch: {epoch} VR Complex Loss per unit: {avg_vr_complex_loss}')
        print("--------------------------------------------------")
    
    print("##### FINISHED TRAINING OF MODEL #####")
    return model, np.vstack((total_losses, mse_losses, tvd_losses, cluster_align_losses))



############ VR-Complex LOSS ############


def vr_complex_loss_clusters(x, latent, labels):
    """
    Compute the topological loss between the input data x and the latent representations latent but only within clusters.
    Args:
    x: Input data tensor of shape (batch_size, num_features)
    latent: Latent representations tensor of shape (batch_size, latent_dim)
    labels: Cluster assignments for each sample
    Returns:
    loss: The topological loss (torch.Tensor)
    """
    # Compute pairwise distance matrices
    x_dists = torch.cdist(x, x)  # Shape: (batch_size, batch_size)
    latent_dists = torch.cdist(latent, latent)

    labels = labels.cpu().numpy()
    # Group indices by label
    label_rows = {}
    for i, label in enumerate(labels):
        label_rows.setdefault(label, []).append(i)

    # Generate pairs of indices within each cluster
    indices = [pair for cluster in label_rows.values() for pair in combinations(cluster, 2)]
    indices = np.array(indices)
    

    # Compute loss using the indices
    loss = topological_loss(x_dists, latent_dists, indices, indices)
    return loss


def vr_complex_loss(x, latent):
    """
    Compute the topological loss between the input data x and the latent representations latent.

    Args:
        x: Input data tensor of shape (batch_size, num_features)
        latent: Latent representations tensor of shape (batch_size, latent_dim)

    Returns:
        loss: The topological loss (torch.Tensor)
    """
    # Compute pairwise distance matrices
    x_dists = torch.cdist(x, x)  # Shape: (batch_size, batch_size)
    latent_dists = torch.cdist(latent, latent)

    # Compute persistence diagrams
    # ret_x = vr_persistence(x_dists, max_dimension=1)
    # ret_latent = vr_persistence(latent_dists, max_dimension=1)

    # pi_X = ret_x[0][0]
    # pi_Z = ret_latent[0][0]

    sorted_indicies_x = get_sorted_indices(x_dists)
    sorted_indicies_latent = get_sorted_indices(latent_dists)

    # Compute the topological loss
    loss = topological_loss(x_dists, latent_dists, sorted_indicies_x, sorted_indicies_latent)

    return loss


def get_sorted_indices(A):
    """
    Given a symmetric 2D PyTorch tensor A, return a list of [row, col] indices,
    sorted in ascending order based on the values at these indices,
    excluding the diagonal entries.
    """
    # Get the indices of the upper triangle, excluding the diagonal
    indices = torch.triu_indices(A.size(0), A.size(1), offset=1).to(device)
    # Extract the values at these indices
    values = A[indices[0], indices[1]]
    # Sort the values and get the indices that would sort the array
    sorted_indices = torch.argsort(values)
    # Reorder the row and column indices accordingly
    sorted_row_indices = indices[0][sorted_indices]
    sorted_col_indices = indices[1][sorted_indices]
    # Combine row and column indices into a list of lists
    sorted_indices_list = [ [row, col] for row, col in zip(sorted_row_indices.tolist(), sorted_col_indices.tolist()) ]
    return np.array(sorted_indices_list)

def topological_loss(A_X, A_Z, pi_X, pi_Z):
    """
    A_X: Distance matrix for input space X (torch.Tensor)
    A_Z: Distance matrix for latent space Z (torch.Tensor)
    pi_X: Persistence pairings for X
    pi_Z: Persistence pairings for Z
    """
    # Subset the distance matrices using the persistence pairings
    # Retrieve the topologically relevant distances using the pairings
    # pi_X = torch.nonzero(A_X.unsqueeze(1) == pi_X[:, 1].unsqueeze(0), as_tuple=False)
    # pi_Z = torch.nonzero(A_Z.unsqueeze(1) == pi_Z[:, 1].unsqueeze(0), as_tuple=False)

    A_X_pi_X = A_X[pi_X[:, 0], pi_X[:, 1]]  # Subset A_X using edges from pi_X
    A_Z_pi_X = A_Z[pi_X[:, 0], pi_X[:, 1]]  # Subset A_Z using same edges from pi_X

    A_Z_pi_Z = A_Z[pi_Z[:, 0], pi_Z[:, 1]]  # Subset A_Z using edges from pi_Z
    A_X_pi_Z = A_X[pi_Z[:, 0], pi_Z[:, 1]]  # Subset A_X using same edges from pi_Z
    
    # Calculate the L2 loss terms (differences between the selected distances)
    L_X_to_Z = torch.mean((A_X_pi_X - A_Z_pi_X) ** 2)
    L_Z_to_X = torch.mean((A_Z_pi_Z - A_X_pi_Z) ** 2)
    
    # Total topological loss
    Lt = 0.5 * (L_X_to_Z + L_Z_to_X)
    
    return Lt

################ SPREAD LOSS ################
def spread_loss(latent_embeddings, labels, min_variance=0.10):
    """
    Computes a spread loss that penalizes clusters whose variance in the latent space is too small.
    Args:
        latent_embeddings (torch.Tensor): Latent space representations of the data.
        labels (torch.Tensor): Cluster assignments for each sample.
        min_variance (float): Minimum allowable variance for each cluster.

    Returns:
        torch.Tensor: The spread loss to ensure variance in latent space is not too small.
    """
    unique_labels = labels.unique()
    spread_loss_value = 0.0
    
    for label in unique_labels:
        # Get the embeddings for the current cluster
        cluster_points = latent_embeddings[labels == label]
        
        if len(cluster_points) > 1:
            # Compute the variance of the points in this cluster
            cluster_variance = torch.var(cluster_points, dim=0)
            
            # Penalize clusters whose variance is too small
            variance_penalty = F.relu(min_variance - cluster_variance).mean()
            spread_loss_value += variance_penalty
    
    return spread_loss_value / len(unique_labels)  # Average across clusters


##################### CLUSTER LOSS #####################
def get_main_cell_pops(data, k):
    gmm = GaussianMixture(n_components=k, random_state=0).fit(data)
    return gmm.means_, gmm.covariances_, gmm.predict(data)


def assign_clusters_and_compute_mse(pred_ae, cluster_centers, cluster_covs, batch_labels):
    """
    Assigns each row in pred_ae to the nearest cluster center, generates random points in the
    neighborhood of the cluster center based on the covariance matrix, and computes the average MSE loss.
    
    Args:
    pred_ae (torch.Tensor): 2D tensor of shape (n_samples, n_features)
    cluster_centers (torch.Tensor): 2D tensor of shape (n_clusters, n_features)
    cluster_covs (torch.Tensor): 3D tensor of shape (n_clusters, n_features, n_features)
    batch_labels (torch.Tensor): 1D tensor of cluster assignments for each sample
    
    Returns:
    torch.Tensor: Scalar tensor of average MSE loss across all samples
    """
    
    # Get the assigned cluster centers and covariance matrices for each sample
    assigned_centers = cluster_centers[batch_labels]

    mean_diff_losses = []
    for batch_label in batch_labels.unique():
        mean_pred = pred_ae[batch_labels == batch_label].mean(dim=0)
        mean_diff_losses.append(torch.mean((mean_pred - assigned_centers[batch_label]) ** 2))

    # Compute simple MSE to cluster centers
    return torch.mean(torch.stack(mean_diff_losses))



def compute_clus_dist_loss(x, pred_ae, cluster_centeres, batch_labels, sinkhorn_distance):
    """
    Compute distances to each cluster center, create histograms for each cluster, and compute the loss.
    """

    x_clus_dist = compute_clus_dist_values(x, cluster_centeres, batch_labels)
    pred_clus_dist = compute_clus_dist_values(pred_ae, cluster_centeres, batch_labels)

    total_loss = 0.0

    for (x_vals, pred_vals) in zip(x_clus_dist, pred_clus_dist):
        max_val = max(max(x_vals), max(pred_vals))

        x_hist = generate_hist(x_vals, num_bins=50, min_val=0.0, max_val=max_val)
        pred_hist = generate_hist(pred_vals, num_bins=50, min_val=0.0, max_val=max_val)

        loss = sinkhorn_distance(x_hist.unsqueeze(0), pred_hist.unsqueeze(0))
        total_loss += loss

    return total_loss



def compute_clus_dist_values(data, cluster_centers, batch_labels):
    """
    Compute the Mahalanobis distance between each point in the batch and the cluster center it is assigned to.
    
    Args:
    data (torch.Tensor): 2D tensor of shape (n_samples, n_features)
    cluster_centers (torch.Tensor): 2D tensor of shape (n_clusters, n_features)
    cluster_covs (torch.Tensor): 3D tensor of shape (n_clusters, n_features, n_features)
    batch_labels (torch.Tensor): 1D tensor of cluster assignments for each sample
    
    Returns:
    tuple: (mahalanobis_distances, histograms)
        - mahalanobis_distances: 1D tensor of Mahalanobis distances for each sample
        - histograms: 2D tensor of histograms for each cluster
    """

    # Compute MSE between each point and its assigned cluster center
    assigned_centers = cluster_centers[batch_labels]
    mse = torch.mean((data - assigned_centers)**2, axis=1)
    
    # Compute histograms for each cluster
    values = []
    for label in range(cluster_centers.shape[0]):
        cluster_samples = mse[batch_labels == label]
        values.append(cluster_samples)

    return values

##################### SPREAD LOSS #####################
def tvd_loss(pred, target, sinkhorn_distance, num_bins=200):
    # Get the number of columns (features)
    num_columns = pred.size(1)

    # Compute the Wasserstein distance for each column
    distances = []
    for i in range(num_columns):
        # Get the predicted and target values for the column
        pred_col = pred[:, i].to(device)
        target_col = target[:, i].to(device)

        # Determine the histogram range for the column
        min_val = min(pred_col.min().item(), target_col.min().item())
        max_val = max(pred_col.max().item(), target_col.max().item())

        # Compute the histograms for both predicted and target values
        pred_hist = generate_hist(pred_col, num_bins=num_bins, min_val=float(min_val), max_val=float(max_val))
        target_hist = generate_hist(target_col, num_bins=num_bins, min_val=float(min_val), max_val=float(max_val))

        # Calculate the Wasserstein distance (Earth Mover's Distance)
        # wasserstein_dist = nn.MSELoss(reduction='sum')(pred_hist, target_hist)
        tvd_dist = sinkhorn_distance(pred_hist.unsqueeze(0), target_hist.unsqueeze(0))
        distances.append(tvd_dist)

    # Compute the mean of the Wasserstein distances across all columns
    mean_tvd_distance = torch.mean(torch.stack(distances))

    return mean_tvd_distance

def generate_hist(feature_values, num_bins, min_val, max_val):
    """
    Generate a histogram from a tensor of values, in a differentiable manner.
    
    Parameters:
    - feature_values: Tensor of feature values (shape: [N]).
    - num_bins: Number of bins in the histogram.
    - min_val: Minimum value in the range of the histogram.
    - max_val: Maximum value in the range of the histogram.

    Returns:
    - Tensor representing the histogram (shape: [num_bins]).
    """
    bin_centre_offset = (max_val - min_val) / (2 *num_bins)
    # Define bin centers
    bin_centers = torch.linspace(min_val + bin_centre_offset, max_val - bin_centre_offset, num_bins).to(feature_values.device)

    # Calculate the bin width
    bin_width = (max_val - min_val) / num_bins
    
    # Compute bin probabilities
    feature_values_expanded = feature_values.unsqueeze(1)

    bin_centers_expanded = bin_centers.unsqueeze(0)
    
    # Softmax-like bin assignment
    distances = torch.abs(feature_values_expanded - bin_centers_expanded)
    bin_probs = torch.exp(-distances / (0.5 * bin_width))
    
    # Normalize bin probabilities
    bin_probs_sum = bin_probs.sum(dim=0)
    histogram = bin_probs_sum / bin_probs_sum.sum()
    
    return histogram



##################### UTILITY FUNCTIONS #####################
def load_data(panel: str, load_full: bool = False) -> np.ndarray:
    """ Load data from a panel
    
    Args:
        panel: The panel to load data from
        
    Returns:
        np.ndarray: The data from the panel
    """
    if (not load_data and os.path.exists(somepath + panel + ".npy")):
        return np.load(somepath + panel + ".npy")

    if (platform.system() == "Windows"):
        full_panel = somepath + panel + "\\"
    else:
        full_panel = somepath + panel + "/"

    # Recursively search for all .fcs files in the directory and subdirectories
    fcs_files = glob.glob(os.path.join(full_panel, '**', '*.fcs'), recursive=True)
    fcs_files_np = []

    if (platform.system() == "Windows"):
        spillover = "C:\\Users\\aksha\\Documents\\ANU\\COMP4550_(Honours)\\Spillovers\\281122_Spillover_Matrix.csv"
    else:
        spillover = "/home/akshat/Documents/281122_Spillover_Matrix.csv"
    
    # Load each .fcs file into fk.Sample and print it
    for fcs_file in fcs_files:
        sample = fk.Sample(fcs_file)
        if "Panel" in full_panel:
            sample.apply_compensation(spillover)
        else:
            sample.apply_compensation(sample.metadata['spill'])
        sample.apply_transform(transform)
        fcs_files_np.append(get_np_array_from_sample(sample, subsample=True))

    res = np.vstack(fcs_files_np)
    np.save(somepath + panel + ".npy", res)
    return res


def get_dataloader(data: np.ndarray, labels: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
    """ Get a DataLoader from a np.ndarray and corresponding labels
    
    Args:
        data: The np.ndarray containing the data
        labels: The np.ndarray containing the labels corresponding to the data
        batch_size: The batch size to use
        
    Returns:
        torch.utils.data.DataLoader: The DataLoader representation of the np.ndarray and labels    
    """
    # Convert data and labels to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # Use long dtype for classification labels
    
    # Create a dataset that contains both the data and the labels
    dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    
    # Return the DataLoader with the dataset, shuffling both data and labels together
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_channel_source(channel: str) -> str:
    """ Get the source of the channel

    Args:
        channel: The channel to get the source of

    Returns:
        str: The source of the channel
    """

    if channel in scatter_channels:
        return 'raw'
    return 'xform'

def get_factor(channel: str) -> float:
    """ Get the factor to divide the channel by

    Args:
        channel: The channel to get the factor for

    Returns:
        float: The factor to divide the channel by
    """

    if channel in scatter_channels:
        return 262144.0
    return 1.0

def get_np_array_from_sample(sample: fk.Sample, subsample: bool) -> np.ndarray:
    """ Get a np.ndarray from a Sample object

    Args:
        sample: The Sample object to convert
        subsample: Whether to subsample the data

    Returns:
        np.ndarray: The np.ndarray representation of the Sample object
    """

    return np.array([
        sample.get_channel_events(sample.get_channel_index(ch), source=get_channel_source(ch), subsample=subsample) / get_factor(ch)
        for ch in all_channels if ch in sample.pnn_labels
    ]).T


##################### MAIN #####################
if __name__ == "__main__":
    train_models = True
    show_result = True
    batches_to_run = ["Panel1"]
    p_values = None
    folder_path = "OC"


    if train_models:
        for directory in directories:
            if directory in batches_to_run:
                print(f"-------- TRAINING FOR {directory} -----------")
   
                x = load_data(directory)
                ref_centres, ref_cov, ref_labels = get_main_cell_pops(x[:, 6:], 13)
                cluster_centres = torch.tensor(ref_centres, dtype=torch.float32).to(device)
                cluseter_cov = torch.tensor(ref_cov, dtype=torch.float32).to(device)

                # data = get_dataloader(x, ref_labels, 8196)
                # model = BNorm_AE(x.shape[1], 3)

                data = get_dataloader(x[:, 6:], ref_labels, 1024)
                model = BNorm_AE_Overcomplete(x.shape[1] - 6, 48)

                # model.load_state_dict(torch.load(f'S_3/3.0_model_{directory}.pt', map_location=device))
                # model = model.to(device)

                model, losses = train_model(model, data, 200, 0.0001, 0.3, cluster_centres, cluseter_cov)
                np.save(f'{folder_path}/losses_{directory}.npy', losses)
                torch.save(model.state_dict(), f'{folder_path}/model_{directory}.pt')
                print(f"-------- FINISHED TRAINING FOR {directory} -----------")
            

    if show_result:
        for directory in directories:
            if directory in batches_to_run:
                print(f"-------- COMPUTING RESULTS FOR {directory} -----------")
                x = load_data(directory)
                num_cols = x.shape[1]

                # model = BNorm_AE(x.shape[1], 3)
                model = BNorm_AE_Overcomplete(x.shape[1] - 6, 48)
                model.load_state_dict(torch.load(f'{folder_path}/model_{directory}.pt', map_location=device))
                model = model.to(device)

                x_tensor = torch.tensor(x[:, 6:], dtype=torch.float32).to(device)

                x_transformed = model(x_tensor)[0].cpu().detach().numpy()  
                x_transformed = np.hstack((x[:, :6], x_transformed))

                np.save(f'./{directory}_x.npy', x_transformed)
                print(f"-------- FINISHED SHOWING RESULTS FOR {directory} -----------")
