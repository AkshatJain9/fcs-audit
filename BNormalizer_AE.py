import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import flowkit as fk
import glob
import os
import matplotlib.pyplot as plt

scatter_channels = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W']
fluro_channels = ['BUV 395-A', 'BUV737-A', 'Pacific Blue-A', 'FITC-A', 'PerCP-Cy5-5-A', 'PE-A', 'PE-Cy7-A', 'APC-A', 'Alexa Fluor 700-A', 'APC-Cy7-A','BV510-A','BV605-A']
new_channels = ['APC-Alexa 750 / APC-Cy7-A', 'Alexa 405 / Pac Blue-A', 'Qdot 605-A']
fluro_channels += new_channels

all_channels = scatter_channels + fluro_channels

dataset = "/home/akshat/Documents/Data/Plate 19635 _CD8"
transform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)


reduce_dims = 2

class BNorm_AE(nn.Module):
    def __init__(self, ch_count):
        super(BNorm_AE, self).__init__()

        # Remove scatter channels
        self.down1 = nn.Linear(in_features=ch_count, out_features=(ch_count - 3))
        self.down2 = nn.Linear(in_features=(ch_count - 3), out_features=(ch_count - 6))

        # Real dimensionality reduction of fluoro channels
        self.down3 = nn.Linear(in_features=(ch_count - 6), out_features=(ch_count - 6 - (reduce_dims / 2)))
        self.down4 = nn.Linear(in_features=(ch_count - 6 - (reduce_dims / 2)), out_features=(ch_count - 6 - reduce_dims))

        # Build up back to fluoro dimensionality
        self.up1 = nn.Linear(in_features=(ch_count - 6 - reduce_dims), out_features=(ch_count - 6 - (reduce_dims / 2)))
        self.up2 = nn.Linear(in_features=(ch_count - 6 - (reduce_dims / 2)), out_features=(ch_count - 6))
        

        self.relu = nn.ReLU(inplace=True)


    def forward(self, input_data):
        x = self.down1(input_data)
        x = self.relu(x)
        x = self.down2(x)
        x = self.relu(x)
        x = self.down3(x)
        x = self.relu(x)
        x = self.down4(x)

        y = self.up1(x)
        y = self.relu(y)
        y = self.up2(y)
        return y
    


def train_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, epoch_count: int, learning_rate: float) -> nn.Module:
        print("##### STARTING TRAINING OF MODEL #####")
        model.train()
        criterion_ae = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for epoch in range(epoch_count):
            total_loss = 0.0
            total_samples = 0
            for batch in data_loader:
                x = batch[0]
                x = x.to(device)

                optimizer.zero_grad()
                pred_ae = model(x)

                loss_ae = criterion_ae(pred_ae, x)
                loss_ae.backward()
                optimizer.step()

                total_loss += loss_ae.item()
                total_samples += x.size(0)

            avg_loss = total_loss / total_samples
            print(f'Epoch: {epoch} Loss per unit: {avg_loss}')
        
        print("##### FINISHED TRAINING OF MODEL #####")
        return model

def load_data(panel: str) -> np.ndarray:
    panel = "/home/akshat/Documents/Data/" + panel + "/"

    # Recursively search for all .fcs files in the directory and subdirectories
    fcs_files = glob.glob(os.path.join(panel, '**', '*.fcs'), recursive=True)

    fcs_files_np = []

    printed = False
    
    # Load each .fcs file into fk.Sample and print it
    for fcs_file in fcs_files:
        sample = fk.Sample(fcs_file)
        if not printed:
            print(sample.pnn_labels)
            printed = True
        sample.apply_transform(transform)
        fcs_files_np.append(get_np_array_from_sample(sample, subsample=False))

    return np.vstack(fcs_files_np)



def get_np_array_from_sample(sample: fk.Sample, subsample: bool) -> np.ndarray:
    """ Get a np.ndarray from a Sample object

    Args:
        sample: The Sample object to convert
        subsample: Whether to subsample the data

    Returns:
        np.ndarray: The np.ndarray representation of the Sample object
    """

    return np.array([
        sample.get_channel_events(sample.get_channel_index(ch), source='raw', subsample=subsample)
        for ch in all_channels if ch in sample.pnn_labels
    ]).T


# x = load_data(dataset)
# print(x.shape)

# /home/akshat/Documents/Data/Plate 19635 _CD8 - 15 channels

somepath = '/home/akshat/Documents/Data/'

# List all directories in the specified path
directories = [d for d in os.listdir(somepath) if os.path.isdir(os.path.join(somepath, d))]

# Print each directory
for directory in directories:
    print("-------------------")
    print("Loading Data for: ", directory)
    x = load_data(directory)
    print(f"Loaded {directory} with shape {x.shape}")
    print("")

    # Determine the number of columns
    num_cols = x.shape[1]
    
    # Create a grid of subplots with num_cols rows and 1 column
    fig, axes = plt.subplots(num_cols, 1, figsize=(6, 4*num_cols))  # Adjust figure size
    
    # Plot histogram for each column in a subplot
    for i in range(num_cols):
        axes[i].hist(x[:, i], bins=200, alpha=0.7)
        axes[i].set_title(f'Column {i+1} Histogram')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure to the Downloads directory
    save_path = os.path.join("/home/akshat/Downloads/", f'{directory}.png')
    plt.savefig(save_path)
    
    print(f"Saved figure as {save_path}")


