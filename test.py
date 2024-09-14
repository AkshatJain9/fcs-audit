import torch

def get_sorted_indices(A):
    """
    Given a symmetric 2D PyTorch tensor A, return a list of [row, col] indices,
    sorted in ascending order based on the values at these indices,
    excluding the diagonal entries.
    """
    # Get the indices of the upper triangle, excluding the diagonal
    indices = torch.triu_indices(A.size(0), A.size(1), offset=1)
    # Extract the values at these indices
    values = A[indices[0], indices[1]]
    # Sort the values and get the indices that would sort the array
    sorted_indices = torch.argsort(values)
    # Reorder the row and column indices accordingly
    sorted_row_indices = indices[0][sorted_indices]
    sorted_col_indices = indices[1][sorted_indices]
    # Combine row and column indices into a list of lists
    sorted_indices_list = [ [row, col] for row, col in zip(sorted_row_indices.tolist(), sorted_col_indices.tolist()) ]
    return sorted_indices_list

# Example usage:
A = torch.tensor([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]], dtype=torch.float)

sorted_indices = get_sorted_indices(A)
print(sorted_indices)

# Epoch: 0 Loss per unit: 0.00012614261969923972
# Epoch: 0 MSE Loss per unit: 4.76522421464324e-06
# Epoch: 0 TVD Loss per unit: 3.2664120371919126e-07
# Epoch: 0 Cluster Alignment Loss per unit: 1.264412285760045e-05
# Epoch: 0 VR Complex Loss per unit: 0.0006196858558058739
# --------------------------------------------------