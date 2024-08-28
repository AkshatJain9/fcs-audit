import numpy as np
import matplotlib.pyplot as plt
import os

def graph_all_losses():
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]  # List all .npy files in the current directory
    
    plt.figure()  # Initialize the figure
    for file in npy_files:
        loss_arr = np.load(file)  # Load each .npy file
        plt.plot(loss_arr, label=file)  # Plot the loss curve with the filename as the label

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per unit over epochs')
    plt.legend()  # Add a legend to differentiate the curves
    plt.show()

graph_all_losses()
