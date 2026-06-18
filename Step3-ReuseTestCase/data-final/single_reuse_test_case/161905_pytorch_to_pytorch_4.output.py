import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

# Setup device
device = 'mps'

# Check if MPS is available to avoid runtime errors on non-Mac machines
if not torch.backends.mps.is_available():
    print("MPS device is not available. Skipping test.")
else:
    # Define a function that uses torch.all, decorated with torch.compile
    # This adapts the original pattern of using @torch.compile to test the similar API
    @torch.compile
    def check_all(tensor: Tensor) -> Tensor:
        return torch.all(tensor)

    # Create test tensors on the MPS device
    # Case 1: All elements are True
    t_all_true = torch.ones(10, device=device).bool()
    
    # Case 2: Some elements are False
    t_some_false = torch.tensor([True, False, True, True], device=device)

    # Case 3: Float tensor (all non-zero)
    t_float_non_zero = torch.randn(5, 5, device=device).abs() + 0.1

    # Execute the compiled function and verify results
    res_true = check_all(t_all_true)
    res_false = check_all(t_some_false)
    res_float = check_all(t_float_non_zero)

    # Assertions to verify correctness
    assert res_true.item() == True, "Expected True for tensor of all True values"
    assert res_false.item() == False, "Expected False for tensor with some False values"
    assert res_float.item() == True, "Expected True for tensor of all non-zero float values"

    print("Test passed: torch.all works correctly with torch.compile on MPS backend.")