import numpy as np
import matplotlib.pyplot as plt
import os
import platform

def graph_all_losses(name):
    npy_files = []
    
    # Walk through current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        # Exclude directories with 'W' in their name
        dirs[:] = [d for d in dirs if 'W' not in d]
        for file in files:
            # Check if the file is a .npy file and contains the given name
            if file.endswith('.npy') and name in file:
                npy_files.append(os.path.join(root, file))
    
    if not npy_files:
        print(f"No .npy files found with name '{name}'")
        return
    
    plt.figure(figsize=(12, 6))  # Initialize the figure with a larger size
    for file in npy_files:
        loss_arr = np.load(file)  # Load each .npy file
        # Use the parent directory name as the label
        label = os.path.basename(os.path.dirname(file))
        plt.plot(loss_arr, label=label)  # Plot the loss curve with the directory name as the label

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss per unit over epochs for files containing "{name}"')
    plt.legend()  # Add a legend to differentiate the curves
    plt.tight_layout()  # Adjust the layout to prevent cutting off labels
    plt.show()


def graph_all_mixed_losses(name):
    npy_files = []
    for root, dirs, files in os.walk('.'):
        # Exclude directories with 'W' in their name
        dirs[:] = [d for d in dirs]
        for file in files:
            # Check if the file is a .npy file and contains the given name
            if file.endswith('.npy') and name in file:
                npy_files.append(os.path.join(root, file))

    if not npy_files:
        print(f"No .npy files found with name '{name}'")
        return
    
    plt.figure(figsize=(12, 6))  # Initialize the figure with a larger size
    for file in npy_files:
        loss_arr = np.load(file)  # Load each .npy file
        label = os.path.basename(os.path.dirname(file))

        print(file)

        if (platform.system() == 'Windows'):
            p_value = float(file.split('\\')[2].split("_")[0]) / 10
        else:
            p_value = float(file.split('/')[2].split("_")[0]) / 10
        
        total_loss = loss_arr[0]
        mse_loss = loss_arr[1] / p_value
        wasserstein_loss = loss_arr[2] / (1 - p_value)

        # Use the parent directory name as the label
        
        # plt.plot(total_loss, label=f'{p_value}-{label} - Total Loss')
        plt.plot(mse_loss, label=f'{p_value}-{label} - MSE Loss')
        # plt.plot(wasserstein_loss, label=f'{p_value}-{label} - Wasserstein Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss per unit over epochs for files containing "{name}"')
    plt.legend()  # Add a legend to differentiate the curves
    plt.ylim(0, 0.00004)  # Set y-axis limits (adjust the values as needed)
    plt.tight_layout()  # Adjust the layout to prevent cutting off labels
    plt.show()
    

graph_all_mixed_losses("Panel1")  # Call the function with the desired name

# Plate 19635
# Plate 27902
# Plate 28332
# Plate 28528
# Plate 29178
# Plate 36841
