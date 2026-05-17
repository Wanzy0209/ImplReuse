import torch
import pytest

def test_jagged_tensor_cat_stack_dim_0_crash():
    """
    Test case for Issue 161812: Crash in jagged tensor stack/cat along dimension 0.
    
    This test reproduces the ValueError thrown when attempting to concatenate or 
    stack jagged nested tensors along dimension 0. The error indicates a schema 
    mismatch where the dispatch function expects 2 arguments but receives 1.
    """
    # Setup: Create a jagged nested tensor
    # The input list contains tensors of different sizes in the first dimension (3 vs 4)
    x = torch.nested.nested_tensor(
        [torch.ones(3, 2, 3), torch.ones(4, 2, 3)], 
        layout=torch.jagged
    )

    # Reproduction 1: torch.cat (default dim=0)
    # Bug: ValueError: NestedTensor cat_default(tensors: any, dim: any): 
    # expected at least 2 arguments and at most 2 arguments, but got: 1 arguments
    with pytest.raises(ValueError, match="expected at least 2 arguments"):
        torch.cat([x, x])

    # Reproduction 2: torch.cat with explicit dim=0
    with pytest.raises(ValueError, match="expected at least 2 arguments"):
        torch.cat([x, x], dim=0)

    # Reproduction 3: torch.stack
    # Bug: Similar ValueError for stack_default
    with pytest.raises(ValueError, match="expected at least 2 arguments"):
        torch.stack([x, x])

    # Reproduction 4: torch.vstack
    # Bug: Similar ValueError
    with pytest.raises(ValueError, match="expected at least 2 arguments"):
        torch.vstack([x, x])

    # Note: The bug report states that passing other dimensions works.
    # This test focuses specifically on the failing case (dim 0).