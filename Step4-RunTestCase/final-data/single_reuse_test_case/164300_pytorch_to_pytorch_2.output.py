import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import functools

# Define a simple model for testing
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

model = TestModel()

# Define a custom pruning method class
# global_unstructured expects a class with PRUNING_TYPE attribute, not a function
class CustomPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        # Store the amount, though global_unstructured handles the threshold calculation
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # For global unstructured, the mask is computed by global_unstructured
        # based on the global threshold. This method is technically not used for the calculation
        # but the class structure is required by the API.
        return default_mask

# Parameters to prune
parameters = ((model.linear, 'weight'),)

# Call the API: torch.nn.utils.prune.global_unstructured
# We pass the class directly and the amount to global_unstructured.
# functools.partial is not suitable here because global_unstructured needs to instantiate
# the class with the amount argument to determine PRUNING_TYPE and configuration.
prune.global_unstructured(
    parameters,
    pruning_method=CustomPruningMethod,
    amount=0.5
)

# Assertions to verify the behavior
assert hasattr(model.linear, 'weight_mask'), "Pruning mask was not created"
assert torch.sum(model.linear.weight_mask == 0) > 0, "No weights were pruned"
assert torch.sum(model.linear.weight_mask == 1) > 0, "All weights were pruned"

print("Test passed.")