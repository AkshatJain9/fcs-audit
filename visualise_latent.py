import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_histogram(panel_np, index, min_val, max_val):
    range = (min_val, max_val)

    hist, bin_edges = np.histogram(panel_np[:, index], bins=200, range=range)
    hist = hist / np.sum(hist)  # Normalize the histogram
    return hist, bin_edges

def plot_histograms(panel_np_1):
    num_channels = panel_np_1.shape[1]
    
    # Create a figure with subplots, one for each channel
    fig, axs = plt.subplots(12, 1, figsize=(8, 2*num_channels))
    
    # If there's only one subplot, axs won't be an array, so wrap it in one
    if num_channels == 1:
        axs = [axs]
    
    for i, ax in enumerate(axs):
        # Get the minimum and maximum values for scaling the histograms
        min_val = np.min(panel_np_1[:, i])
        max_val = np.max(panel_np_1[:, i])
        
        # Generate histogram for the current channel of panel_np_1
        hist = generate_histogram(panel_np_1, i, min_val, max_val)
        
        # Plot the histogram on the current axis
        ax.plot(hist)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


def plot_fluoro_hist_compare(data_list, names=None, idx=None, file_name=None):
    """
    Plots histograms comparing fluorescence data across multiple datasets.

    Parameters:
        data_list (list of numpy arrays): A list of datasets to be compared.
        names (list of str): Names used for labeling the datasets.
        idx (int or list of ints): Index or indices of columns to plot. If None, plot all columns starting from index 6.
        synth_batch (bool): If True, use synthetic batch labels; otherwise, use default labels.
        file_name (str): Name used for saving the plot.
    """
    num_datasets = len(data_list)

    if num_datasets < 2:
        print("Error: At least two datasets are required for comparison.")
        return

    num_channels = data_list[0].shape[1]

    # Ensure all datasets have the same number of channels
    for data in data_list:
        if data.shape[1] != num_channels:
            print("Error: All datasets must have the same number of channels.")
            return

    # Determine the indices of channels to plot
    if idx is not None:
        # If idx is a single integer, convert it to a list
        if isinstance(idx, int):
            channels_to_plot = [idx]
        else:
            channels_to_plot = idx  # Assume idx is iterable
    else:
        # Plot all channels starting from index 6
        channels_to_plot = list(range(6, num_channels))

    remaining_channels = len(channels_to_plot)
    if remaining_channels == 0:
        print("Error: No channels to plot.")
        return

    fig2, axs2 = plt.subplots(remaining_channels, 1, figsize=(8, 2 * remaining_channels))

    # If only one subplot, axs2 may not be a list, so make it into a list
    if remaining_channels == 1:
        axs2 = [axs2]

    # Choose labels based on synth_batch flag
    colors = sns.color_palette("bright", len(data_list))

    for channel_index, ax in zip(channels_to_plot, axs2):
        # Determine the min and max values across all datasets for the current channel
        min_val = np.min([np.min(data[:, channel_index]) for data in data_list])
        max_val = np.max([np.max(data[:, channel_index]) for data in data_list])

        # Plot histograms for each dataset
        for dataset_idx, data in enumerate(data_list):
            if names is not None:
                data_name = names[dataset_idx]
            else:
                data_name = f'Dataset {dataset_idx+1}'

            hist, bin_edges = generate_histogram(data, channel_index, min_val, max_val)
            ax.plot(bin_edges[:-1], hist, label=data_name, color=colors[dataset_idx])

        # Set axis labels using the provided labels
        ax.set_xlabel("Latent Variable Value")
        ax.set_ylabel('Frequency (Relative)')
        ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(f'latent_{file_name}.png')
    else:
        plt.savefig(f'latent.png')
    plt.close()


if __name__ == "__main__":
    p1_latent = np.load('Panel1_latent.npy')
    p2_latent = np.load('target_batch_encoded_Panel2.npy')
    p2_latent_shifted = np.load('target_batch_shifted_Panel2.npy')

    plot_fluoro_hist_compare([p1_latent, p2_latent, p2_latent_shifted], idx=19, names=['Panel 1', 'Panel 2', 'Panel 2 Shifted'])