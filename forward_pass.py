import torch

# 1. Define the inputs, weights, and bias as PyTorch tensors
# Using the values from the previous perceptron calculation
X = torch.tensor([2.0, 3.0])      # Inputs
W = torch.tensor([0.5, -1.0])     # Weights
b = torch.tensor(1.5)             # Bias

# 2. Calculate the weighted sum
# torch.dot multiplies the matching pairs and sums them up, then we add the bias
z = torch.dot(X, W) + b

# 3. Apply the Step activation function
# This checks if z > 0 (returns True/False) and converts it to 1.0 or 0.0
output = (z > 0).float()

print(f"Weighted Sum (z): {z.item()}")
print(f"Final Output: {output.item()}")