import torch
import torch.distributions as dist

def test_compile_does_not_modify_global_distribution_validation():
    """
    Test that torch.compile does not inadvertently change the global
    torch.distributions.Distribution validation state.
    
    Bug Report Reference: Issue #167064
    Description: Calling torch.compile (specifically in certain contexts) 
    was found to call torch.distributions.Distribution.set_default_validate_args(False),
    which affects global state.
    """
    # Save the original state to restore it later
    original_state = dist.Distribution._default_validate_args

    try:
        # Explicitly set validation to True to detect if it gets flipped to False
        dist.Distribution.set_default_validate_args(True)
        assert dist.Distribution._default_validate_args is True, "Setup failed: Could not set validation to True"

        # Define a simple function to compile
        def simple_model(x):
            return x + 1

        # Call torch.compile (the API under test)
        compiled_model = torch.compile(simple_model)

        # Execute the compiled model to trigger the compilation path
        input_tensor = torch.randn(2, 2)
        compiled_model(input_tensor)

        # Verify that the global state has not been changed by torch.compile
        assert dist.Distribution._default_validate_args is True, \
            "torch.compile modified the global distribution validation state to False"

    finally:
        # Restore the original state
        dist.Distribution.set_default_validate_args(original_state)

if __name__ == "__main__":
    test_compile_does_not_modify_global_distribution_validation()
    print("Test passed.")