import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_histogram(panel_np, index, min_val, max_val):
    range = (min_val, max_val)
    hist, bin_edges = np.histogram(panel_np[:, index], bins=200, range=range)
    hist = hist / np.sum(hist)  # Normalize the histogram
    return hist, bin_edges

def plot_fluoro_hist_compare(data_list, names=None, idx=None, file_name=None):
    """
    Plots histograms comparing fluorescence data across multiple datasets.
    Splits the images so that each image contains up to 9 histograms.

    Parameters:
        data_list (list of numpy arrays): A list of datasets to be compared.
        names (list of str): Names used for labeling the datasets.
        idx (int or list of ints): Index or indices of columns to plot. If None, plot all channels.
        file_name (str): Base name used for saving the plots.
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
        # Plot all channels
        channels_to_plot = list(range(num_channels))

    remaining_channels = len(channels_to_plot)
    if remaining_channels == 0:
        print("Error: No channels to plot.")
        return

    # Split channels_to_plot into batches of size 9
    batch_size = 6
    channel_batches = [channels_to_plot[i:i + batch_size] for i in range(0, len(channels_to_plot), batch_size)]

    # Choose colors for datasets
    colors = sns.color_palette("bright", len(data_list))

    for batch_num, channel_batch in enumerate(channel_batches, start=1):
        num_subplots = len(channel_batch)
        fig2, axs2 = plt.subplots(num_subplots, 1, figsize=(8, 2 * num_subplots))

        # If only one subplot, axs2 may not be a list, so make it into a list
        if num_subplots == 1:
            axs2 = [axs2]

        for ax_idx, (channel_index, ax) in enumerate(zip(channel_batch, axs2)):
            # Determine the min and max values across all datasets for the current channel
            min_val = np.min([np.min(data[:, channel_index]) for data in data_list])
            max_val = np.max([np.max(data[:, channel_index]) for data in data_list])

            # Plot histograms for each dataset
            for dataset_idx, data in enumerate(data_list):
                if names is not None:
                    data_name = names[dataset_idx]
                else:
                    data_name = f'Dataset {dataset_idx + 1}'

                hist, bin_edges = generate_histogram(data, channel_index, min_val, max_val)
                ax.plot(bin_edges[:-1], hist, label=data_name, color=colors[dataset_idx])

            # Set axis labels
            ax.set_xlabel(f"Latent Variable Value (Channel {channel_index})")
            ax.set_ylabel('Frequency (Relative)')
            ax.legend()

        # Adjust layout and save the figure
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(f'{file_name}_batch{batch_num}.png')
        else:
            plt.savefig(f'latent_batch{batch_num}.png')
        plt.close()

if __name__ == "__main__":
    p1_latent = np.load('ref.npy')
    p2_latent = np.load('Plate 27902_N.npy')

    plot_fluoro_hist_compare(
        [p1_latent, p2_latent],
        names=['Plate 19635_CD8', 'Plate 27902_N'],
        file_name='latent_comparison'
    )
