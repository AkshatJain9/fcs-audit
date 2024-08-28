import numpy as np
import matplotlib.pyplot as plt
import os

def graph_all_losses(name):
    npy_files = []
    
    # Walk through current directory and subdirectories
    for root, dirs, files in os.walk('.'):
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


graph_all_losses("Plate 39630")

# Plate 19635
# Plate 27902
# Plate 28332
# Plate 28528
# Plate 29178
# Plate 36841
