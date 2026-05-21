import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import L1Unstructured

def test_global_unstructured_pruning():
    """
    Test case for torch.nn.utils.prune.global_unstructured.
    This test verifies that global pruning correctly applies masks across multiple parameters
    and achieves the desired global sparsity.
    """
    
    # Define a simple model with multiple layers to prune globally
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10) # Assuming input size 8x8 feature map

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleModel()
    
    # Parameters to prune: (module, name) tuples
    parameters_to_prune = [
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc, 'weight'),
    ]

    # Calculate initial number of parameters
    initial_params = sum(m.weight.nelement() for m, _ in parameters_to_prune)

    # Apply global unstructured pruning
    # We aim to prune 20% of the connections globally across all selected parameters
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=L1Unstructured,
        amount=0.2
    )

    # Verify that masks were created and original parameters are stored
    for module, name in parameters_to_prune:
        assert hasattr(module, name + '_mask'), f"Mask not found for {name}"
        assert hasattr(module, name + '_orig'), f"Original param not found for {name}"
        
        # Verify mask is a tensor of 0s and 1s
        mask = getattr(module, name + '_mask')
        assert torch.all((mask == 0) | (mask == 1)), "Mask contains values other than 0 and 1"

    # Verify global sparsity
    # Count zeros in all pruned parameters
    current_zeros = 0
    for module, name in parameters_to_prune:
        weight = getattr(module, name)
        current_zeros += torch.sum(weight == 0).item()

    calculated_sparsity = current_zeros / initial_params
    
    # Assert that the sparsity is close to the target amount (0.2)
    # We allow a small tolerance due to rounding
    assert abs(calculated_sparsity - 0.2) < 0.01, \
        f"Expected sparsity ~0.2, but got {calculated_sparsity}"

    print("Test passed: Global unstructured pruning verified successfully.")

if __name__ == "__main__":
    test_global_unstructured_pruning()