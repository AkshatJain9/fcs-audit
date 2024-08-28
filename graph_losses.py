import numpy as np
import matplotlib.pyplot as plt

def graph_losses(directory: str):
    loss_arr = np.load(f'losses_{directory}.npy')
    plt.plot(loss_arr)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per unit over epochs')
    plt.show()

graph_losses('Panel1')