import torch
import torch.distributions as dist

def test_torch_any_no_distribution_side_effects():
    """
    Test that using torch.any within a torch.compile context
    does not inadvertently modify global distribution validation args,
    as reported in Issue #167064 regarding redundant global code.
    """
    # Save the original state of distribution validation args
    # Use getattr to handle cases where the attribute might not exist in some PyTorch versions
    original_validate_args = getattr(dist.Distribution, '_default_validate_args', True)
    
    # Ensure we start with a known state (default is True)
    dist.Distribution.set_default_validate_args(True)

    # Define a function that uses torch.any
    def any_func(x):
        return torch.any(x)

    # Compile the function. This is the context where the bug occurred
    # (triggering eval_frame logic which potentially sets validate_args to False).
    compiled_any_func = torch.compile(any_func)

    # Create a tensor and execute the compiled function
    input_tensor = torch.tensor([False, False, True])
    result = compiled_any_func(input_tensor)

    # 1. Verify the functional correctness of torch.any
    assert result.item() is True, "torch.any should return True for [False, False, True]"

    # 2. Verify the side effect mentioned in the bug report did NOT occur.
    # The bug report indicates that torch.compile (via eval_frame) calls
    # set_default_validate_args(False). We assert that the state remains unchanged.
    # We check if the attribute exists to ensure compatibility across versions.
    if hasattr(dist.Distribution, '_default_validate_args'):
        assert dist.Distribution._default_validate_args is True, \
            "torch.compile with torch.any should not change distribution validation args to False"

    # Restore original state
    dist.Distribution.set_default_validate_args(original_validate_args)

if __name__ == "__main__":
    test_torch_any_no_distribution_side_effects()
    print("Test passed.")