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

p = 0.8

# class BNorm_AE(nn.Module):
#     def __init__(self, ch_count, reduce_dims):
#         super(BNorm_AE, self).__init__()

#         # Remove scatter channels
#         self.down1 = nn.Linear(in_features=ch_count, out_features=(ch_count - 3))
#         self.down2 = nn.Linear(in_features=(ch_count - 3), out_features=(ch_count - 6))

#         # Real dimensionality reduction of fluoro channels
#         self.down3 = nn.Linear(in_features=(ch_count - 6), out_features=(ch_count - 6 - (reduce_dims // 2)))
#         self.down4 = nn.Linear(in_features=(ch_count - 6 - (reduce_dims // 2)), out_features=(ch_count - 6 - reduce_dims))

#         # Build up back to fluoro dimensionality
#         self.up1 = nn.Linear(in_features=(ch_count - 6 - reduce_dims), out_features=(ch_count - 6 - (reduce_dims // 2)))
#         self.up2 = nn.Linear(in_features=(ch_count - 6 - (reduce_dims // 2)), out_features=(ch_count - 6))
        

#         self.relu = nn.ReLU(inplace=True)


#     def forward(self, input_data):
#         x = self.down1(input_data)
#         x = self.relu(x)
#         x = self.down2(x)
#         x = self.relu(x)
#         x = self.down3(x)
#         x = self.relu(x)
#         x = self.down4(x)

#         y = self.up1(x)
#         y = self.relu(y)
#         y = self.up2(y)
#         return y


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


def ks_loss(pred, target):
    loss = 0
    for i in range(pred.shape[1]):
        pred_col = pred[:, i]
        target_col = target[:, i]
        # Use a differentiable approximation of KS distance
        pred_cdf = torch.cumsum(torch.sort(pred_col)[0], dim=0) / pred_col.shape[0]
        target_cdf = torch.cumsum(torch.sort(target_col)[0], dim=0) / target_col.shape[0]
        ks_distance = torch.max(torch.abs(pred_cdf - target_cdf))
        loss += ks_distance
    return loss / pred.shape[1]  # Average over all columns

def train_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, epoch_count: int, learning_rate: float) -> np.ndarray:
    print("##### STARTING TRAINING OF MODEL #####")
    model.train()
    criterion_ae = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    losses = []
    
    for epoch in range(epoch_count):
        total_loss = 0.0
        total_samples = 0
        
        for batch in data_loader:
            x = batch[0]
            x = x.to(device)
            optimizer.zero_grad()
            pred_ae = model(x)
            loss_ae = criterion_ae(pred_ae, x[:, 6:])
            loss_ae.backward()
            optimizer.step()
            
            total_loss += loss_ae.item()
            total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples
        losses.append(avg_loss)
        
        print(f'Epoch: {epoch} Loss per unit: {avg_loss}')
    
    print("##### FINISHED TRAINING OF MODEL #####")
    return model, np.array(losses)

def load_data(panel: str) -> np.ndarray:
    if (platform.system() == "Windows"):
        panel = somepath + panel + "\\"
    else:
        panel = somepath + panel + "/"

    # Recursively search for all .fcs files in the directory and subdirectories
    fcs_files = glob.glob(os.path.join(panel, '**', '*.fcs'), recursive=True)

    fcs_files_np = []

    printed = False

    if (platform.system() == "Windows"):
        spillover = "C:\\Users\\aksha\\Documents\\ANU\\COMP4550_(Honours)\\Spillovers\\281122_Spillover_Matrix.csv"
    else:
        spillover = "/home/akshat/Documents/281122_Spillover_Matrix.csv"
    
    # Load each .fcs file into fk.Sample and print it
    for fcs_file in fcs_files:
        sample = fk.Sample(fcs_file)
        if "Panel" in panel:
            sample.apply_compensation(spillover)
        else:
            sample.apply_compensation(sample.metadata['spill'])
        # if not printed:
        #     print(sample.pnn_labels)
        #     printed = True
        sample.apply_transform(transform)
        fcs_files_np.append(get_np_array_from_sample(sample, subsample=True))

    return np.vstack(fcs_files_np)


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


if (platform.system() == "Windows"):
    somepath = 'C:\\Users\\aksha\\Documents\\ANU\\COMP4550_(Honours)\\Data\\'
else:
    somepath = '/home/akshat/Documents/Data/'

# List all directories in the specified path
directories = [d for d in os.listdir(somepath) if os.path.isdir(os.path.join(somepath, d))]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print each directory
for directory in directories:
    if directory == "Plate 29178":
        
        print("-------------------")
        print("Loading Data for: ", directory)
        x = load_data(directory)
        # data = get_dataloader(x, 1024)
        print(x.shape)
        

        # for num in [3,4,5,6]:
        #     model = BNorm_AE(x.shape[1], num)
        #     model, losses = train_model(model, data, 200, 0.0001)
        #     np.save(f'{num}_3/losses_{directory}.npy', losses)
        #     print("Saving Model for: ", directory)
        #     torch.save(model.state_dict(), f'{num}_3/model_{directory}.pt')

        nn_shape = 3

        model = BNorm_AE(x.shape[1], nn_shape)
        model.load_state_dict(torch.load(f'{nn_shape}/model_{directory}.pt', map_location=device))


        x_transformed = model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy()
        x_transformed = np.hstack((x[:, :6], x_transformed))

        # Determine the number of columns
        num_cols = x.shape[1]
        
        # Create a grid of subplots with num_cols rows and 1 column
        fig, axes = plt.subplots(num_cols, 1, figsize=(6, 4*num_cols))  # Adjust figure size
        
        # Plot histogram for each column in a subplot
        for i in range(x.shape[1]):
            axes[i].hist(x[:, i], bins=200, alpha=0.7)
            axes[i].hist(x_transformed[:, i], bins=200, alpha=0.7, label='Transformed', color='red')
            axes[i].set_title(f'Column {i+1} Histogram')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure to the Downloads directory
        # save_path = os.path.join("/home/akshat/Downloads/", f'{directory}.png')
        # plt.savefig(save_path)
        plt.show()
        
        






    


