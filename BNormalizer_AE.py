import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import flowkit as fk
import glob
import os
import matplotlib.pyplot as plt
import platform

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

def train_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, epoch_count: int, learning_rate: float, p: float) -> np.ndarray:
    print("##### STARTING TRAINING OF MODEL #####")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    losses = []
    mse_losses = []
    wasserstein_losses = []
    
    for epoch in range(epoch_count):
        total_loss = 0.0
        total_samples = 0

        total_mse_loss = 0.0
        total_wasserstein_loss = 0.0
        
        for batch in data_loader:
            x = batch[0]
            x = x.to(device)
            optimizer.zero_grad()
            pred_ae = model(x)

            mse_loss_ae = p * mse_loss(pred_ae, x[:, 6:])
            wasserstein_loss = (1 - p) * wasserstein_distance(pred_ae, x[:, 6:])

            loss_ae = mse_loss_ae + wasserstein_loss
            loss_ae.backward()
            optimizer.step()
            
            total_loss += loss_ae.item()
            total_samples += x.size(0)

            total_mse_loss += mse_loss_ae.item()
            total_wasserstein_loss += wasserstein_loss.item()
        
        avg_loss = total_loss / total_samples
        avg_mse_loss = total_mse_loss / total_samples
        avg_wasserstein_loss = total_wasserstein_loss / total_samples
        losses.append(avg_loss)
        mse_losses.append(avg_mse_loss)
        wasserstein_losses.append(avg_wasserstein_loss)
        
        print(f'Epoch: {epoch} Loss per unit: {avg_loss}')
        print(f'Epoch: {epoch} MSE Loss per unit: {avg_mse_loss}')
        print(f'Epoch: {epoch} Wasserstein Loss per unit: {avg_wasserstein_loss}')
        print("--------------------------------------------------")
    
    print("##### FINISHED TRAINING OF MODEL #####")
    return model, np.vstack((losses, mse_losses, wasserstein_losses))


def wasserstein_distance(pred, target, num_bins=200):
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
        wasserstein_dist = nn.MSELoss(reduction='sum')(pred_hist, target_hist)
        distances.append(wasserstein_dist)

    # Compute the mean of the Wasserstein distances across all columns
    mean_wasserstein_distance = torch.mean(torch.stack(distances))

    return mean_wasserstein_distance

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


def load_data(panel: str) -> np.ndarray:
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


def get_dataloader(data: np.ndarray, batch_size: int) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
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




train_models = True
if train_models:
    for directory in directories:
        if "Panel1" in directory:
            
            print("-------------------")
            print("Loading Data for: ", directory)
            x = load_data(directory)
            data = get_dataloader(x, 1024)
            print(x.shape)
            

            for p in [0.7]:
                model = BNorm_AE(x.shape[1], 3)
                model, losses = train_model(model, data, 1000, 0.0001, p)
                np.save(f'S_3/{p * 10}_losses_{directory}.npy', losses)
                print("Saving Model for: ", directory)
                torch.save(model.state_dict(), f'S_3/{p * 10}_model_{directory}.pt')

            # for num in [3,4,5,6]:
            #     model = BNorm_AE(x.shape[1], num)
            #     model, losses = train_model(model, data, 200, 0.0001)
            #     np.save(f'{num}_3/losses_{directory}.npy', losses)
            #     print("Saving Model for: ", directory)
            #     torch.save(model.state_dict(), f'{num}_3/model_{directory}.pt')


# Graph the losses
show_result = True
if show_result:
    directory = "Panel1"
    print("-------------------")
    print("Loading Data for: ", directory)
    x = load_data(directory)
    num_cols = x.shape[1]

    nn_shape = 3

    for p in [0.7]:
        print("P: ", p)
        # Load the model
        model = BNorm_AE(x.shape[1], nn_shape)
        model.load_state_dict(torch.load(f'S_{nn_shape}/{p * 10}_model_{directory}.pt', map_location=device))
        
        # Move the model to the correct device
        model = model.to(device)

        # Convert the data to a tensor and move it to the same device
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        
        # Transform the data using the model and move it back to the CPU for further processing
        x_transformed = model(x_tensor).cpu().detach().numpy()
        
        # Concatenate the transformed data with the first 6 columns of the original data
        x_transformed = np.hstack((x[:, :6], x_transformed))

        np.save("transformed_data.npy", x_transformed)

        # mse_loss = nn.MSELoss()
        # loss = mse_loss(torch.tensor(x_transformed, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
        # print("MSE Loss: ", loss.item())

        # wasserstein_loss = wasserstein_distance(torch.tensor(x_transformed, dtype=torch.float32), torch.tensor(x, dtype=torch.float32))
        # print("Wasserstein Loss: ", wasserstein_loss.item())

        # Select two random columns
        # random_cols = np.random.choice(range(6, num_cols), 2, replace=False)

        # # Scatter plot for the original data
        # plt.figure(figsize=(12, 5))

        # plt.subplot(1, 2, 1)
        # plt.scatter(x[:, random_cols[0]], x[:, random_cols[1]], alpha=0.5)
        # plt.title(f'Scatter Plot for Original Data (columns {random_cols[0]} vs {random_cols[1]})')
        # plt.xlabel(f'Column {random_cols[0]}')
        # plt.ylabel(f'Column {random_cols[1]}')

        # # Scatter plot for the transformed data
        # plt.subplot(1, 2, 2)
        # plt.scatter(x_transformed[:, random_cols[0]], x_transformed[:, random_cols[1]], alpha=0.5)
        # plt.title(f'Scatter Plot for Transformed Data (columns {random_cols[0]} vs {random_cols[1]})')
        # plt.xlabel(f'Column {random_cols[0]}')
        # plt.ylabel(f'Column {random_cols[1]}')

        # plt.tight_layout()
        # plt.show()

        # assert False
    

        # # Determine the number of columns
        # num_cols = x.shape[1]

        # # Create a grid of subplots with num_cols rows and 1 column
        # fig, axes = plt.subplots(num_cols, 1, figsize=(8, 5 * num_cols))  # Increase figure size

        # # Plot histogram for each column in a subplot
        # for i in range(num_cols):
        #     axes[i].hist(x[:, i], bins=200, alpha=0.7)
        #     axes[i].hist(x_transformed[:, i], bins=200, alpha=0.7, label='Transformed', color='red')
        #     axes[i].set_title(f'Column {i+1} Histogram')
        #     axes[i].set_xlabel('Value')
        #     axes[i].set_ylabel('Frequency')

        #     # Center the plot range around the original data
        #     min_val, max_val = np.min(x[:, i]), np.max(x[:, i])
        #     axes[i].set_xlim(min_val - 0.1 * abs(max_val - min_val), max_val + 0.1 * abs(max_val - min_val))
        #     axes[i].set_ylim(0, 15000)  # Add a bit of padding above


        # # Adjust layout with more vertical space
        # plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=10.0)  # Increase padding between plots
        # plt.show()
        


        






    


