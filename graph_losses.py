import numpy as np
import matplotlib.pyplot as plt
import os
import platform

def contains_letters(input_string):
    return any(char.isalpha() for char in input_string)

def graph_all_losses(name):
    npy_files = []
    
    # Walk through current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        # Exclude directories with 'W' in their name
        dirs[:] = [d for d in dirs if not contains_letters(d) and '_' in d]
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
        label = "f - " + os.path.basename(os.path.dirname(file))[0]
        plt.plot(loss_arr, label=label)  # Plot the loss curve with the directory name as the label

    plt.ylim(0, 3e-5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss per unit over epochs for files containing "{name}"')
    plt.legend()
    plt.tight_layout()
    plt.show()


def graph_all_mixed_losses(name):
    npy_files = []
    for root, dirs, files in os.walk('.'):
        # Exclude directories with 'W' in their name
        dirs[:] = [d for d in dirs if 'S_3' in d]
        for file in files:
            # Check if the file is a .npy file and contains the given name
            if file.endswith('.npy') and name in file:
                npy_files.append(os.path.join(root, file))

    if not npy_files:
        print(f"No .npy files found with name '{name}'")
        return
    
    plt.figure(figsize=(12, 6))  # Initialize the figure with a larger size
    for file in npy_files:
        print(file)
        if ("./S_3/losses_Panel1.npy" != file):
            continue
        
        loss_arr = np.load(file)  # Load each .npy file
        if (loss_arr.shape[1] > 200):
            continue
        label = os.path.basename(os.path.dirname(file))

        # if (platform.system() == 'Windows'):
        #     p_value = float(file.split('\\')[2].split("_")[0]) / 10
        # else:
        #     p_value = float(file.split('/')[2].split("_")[0]) / 10

        p_value = 0.3
        
        
        mse_loss = loss_arr[1] / 0.27
        tvd_loss = loss_arr[2] / 0.63
        total_loss = mse_loss + tvd_loss

        if (len(loss_arr) > 3):
            cluster_align_loss = loss_arr[3] / 0.1
            total_loss += cluster_align_loss
            plt.plot(cluster_align_loss, label=f'{label} - Topological Loss')

        # Use the parent directory name as the label
        
        plt.plot(total_loss, label=f'{label} - Total Loss, p={p_value}')
        plt.plot(mse_loss, label=f'{label} - MSE Loss, p={p_value}')
        plt.plot(tvd_loss, label=f'{label} - Histogram Loss, p={p_value}')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss per unit over epochs for files containing "{name}"')
    plt.legend()  # Add a legend to differentiate the curves
    plt.ylim(0, 0.0001)  # Set y-axis limits (adjust the values as needed)
    plt.tight_layout()  # Adjust the layout to prevent cutting off labels
    plt.show()
    

graph_all_mixed_losses("Panel1")  # Call the function with the desired name

# Plate 19635
# Plate 27902
# Plate 28332
# Plate 28528
# Plate 29178
# Plate 36841
