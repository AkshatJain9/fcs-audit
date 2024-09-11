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
import torch.distributions as dist


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
        return y
    
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
    
    for epoch in range(epoch_count):
        total_loss = 0.0
        total_samples = 0
        total_mse_loss = 0.0
        total_tvd_loss = 0.0
        total_cluster_align_loss = 0.0
        
        for batch in data_loader:
            x = batch[0]
            x = x.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_ae = model(x)
            
            # Calculate MSE and Sinkhorn loss
            mse_loss_ae = mse_loss(pred_ae, x[:, 6:])
            tvd_loss_val = tvd_loss(pred_ae, x[:, 6:], sinkhorn_distance)
            cluster_align_loss = assign_clusters_and_compute_mse(pred_ae, cluster_centres, cluster_cov, batch[1])
            
            # Total loss
            total_loss_ae = 0.9 * (0.3 * mse_loss_ae + 0.7 * tvd_loss_val) + 0.1 * cluster_align_loss
            total_loss_ae.backward()
            optimizer.step()
            
            # Track losses
            total_loss += total_loss_ae.item()
            total_samples += x.size(0)
            total_mse_loss += mse_loss_ae.item()
            total_tvd_loss += tvd_loss_val.item()
            total_cluster_align_loss += cluster_align_loss.item()
        
        # Calculate average losses
        avg_loss = total_loss / total_samples
        avg_mse_loss = total_mse_loss / total_samples
        avg_tvd_loss = total_tvd_loss / total_samples
        avg_cluster_align_loss = total_cluster_align_loss / total_samples
        
        # Append losses to lists
        total_losses.append(avg_loss)
        mse_losses.append(avg_mse_loss)
        tvd_losses.append(avg_tvd_loss)
        cluster_align_losses.append(avg_cluster_align_loss)
        
        print(f'Epoch: {epoch} Loss per unit: {avg_loss}')
        print(f'Epoch: {epoch} MSE Loss per unit: {avg_mse_loss}')
        print(f'Epoch: {epoch} TVD Loss per unit: {avg_tvd_loss}')
        print(f'Epoch: {epoch} Cluster Alignment Loss per unit: {avg_cluster_align_loss}')
        print("--------------------------------------------------")
    
    print("##### FINISHED TRAINING OF MODEL #####")
    return model, np.vstack((total_losses, mse_losses, tvd_losses, cluster_align_losses))



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
    assigned_covs = cluster_covs[batch_labels]

    # Step 1: Compute Cholesky decomposition of covariance matrices for faster sampling
    L = torch.linalg.cholesky(assigned_covs)  # Shape: (n_samples, n_features, n_features)

    # Step 2: Sample from standard normal distribution and transform with Cholesky
    z = torch.randn(pred_ae.shape, device=pred_ae.device)  # Shape: (n_samples, n_features)
    
    # Step 3: Generate random points using Cholesky decomposition
    random_points = assigned_centers + torch.bmm(L, z.unsqueeze(-1)).squeeze(-1)  # Shape: (n_samples, n_features)

    # Step 4: Compute the MSE loss between pred_ae and the sampled random points
    mse_diff = torch.mean((pred_ae - random_points) ** 2, dim=1)
    
    # Compute the average loss across all samples
    total_loss = mse_diff.mean()
    
    return total_loss

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
    bin_probs = torch.exp(-distances / (2 * bin_width))
    
    # Normalize bin probabilities
    bin_probs_sum = bin_probs.sum(dim=0)
    histogram = bin_probs_sum / bin_probs_sum.sum()
    
    return histogram



##################### UTILITY FUNCTIONS #####################
def load_data(panel: str) -> np.ndarray:
    """ Load data from a panel
    
    Args:
        panel: The panel to load data from
        
    Returns:
        np.ndarray: The data from the panel
    """
    if (os.path.exists(somepath + panel + ".npy")):
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
    folder_path = "F_3"


    if train_models:
        for directory in directories:
            if directory in batches_to_run:
                print(f"-------- TRAINING FOR {directory} -----------")
   
                x = load_data(directory)
                ref_centres, ref_cov, ref_labels = get_main_cell_pops(x[:, 6:], 7)
                cluster_centres = torch.tensor(ref_centres, dtype=torch.float32).to(device)
                cluseter_cov = torch.tensor(ref_cov, dtype=torch.float32).to(device)

                data = get_dataloader(x, ref_labels, 512)

                model = BNorm_AE(x.shape[1], 3)

                model.load_state_dict(torch.load(f'S_3/3.0_model_{directory}.pt', map_location=device))
                model = model.to(device)

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

                model = BNorm_AE(x.shape[1], 3)
                model.load_state_dict(torch.load(f'{folder_path}/model_{directory}.pt', map_location=device))
                model = model.to(device)

                x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

                x_transformed = model(x_tensor).cpu().detach().numpy()  
                x_transformed = np.hstack((x[:, :6], x_transformed))

                np.save(f'./{directory}_x.npy', x_transformed)
                print(f"-------- FINISHED SHOWING RESULTS FOR {directory} -----------")
