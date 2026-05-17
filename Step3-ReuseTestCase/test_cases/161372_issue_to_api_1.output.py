import torch
import pytest

def test_compile_dynamic_hann_window():
    """
    Test case for torch.compile regression with dynamic shapes.
    
    This test leverages torch.hann_window (the similar API) to reproduce
    the logic of the reported bug where torch.compile fails due to 
    tensor size mismatches (e.g., expected 77, actual 78) hitting the 
    recompilation limit.
    
    The original bug involved an LLM where KV cache sizes changed dynamically.
    Here, we simulate dynamic tensor generation using torch.hann_window,
    which produces a tensor of size (window_length,).
    """
    
    # Define a function that uses the similar API (torch.hann_window)
    # with a dynamic size parameter.
    def dynamic_window_model(window_length):
        # torch.hann_window creates a tensor of size (window_length,)
        # This mimics the behavior of creating tensors with dynamic sequence lengths
        # found in the original LLM bug report.
        window = torch.hann_window(window_length)
        return window

    # Compile the model. 
    # In the bug report, dynamic=True or default settings led to recompilation issues.
    # We test with dynamic=True to ensure the compiler handles varying sizes.
    compiled_model = torch.compile(dynamic_window_model, dynamic=True)

    # The bug report specifically mentioned a mismatch between size 77 and 78.
    # We test a sequence of sizes to ensure the compiler generalizes correctly
    # and does not hit the recompilation limit (config.recompile_limit).
    test_sizes = [10, 20, 77, 78, 79, 100]

    for size in test_sizes:
        try:
            result = compiled_model(size)
            
            # Assertion to verify the output shape matches the input size
            assert result.shape[0] == size, \
                f"Shape mismatch for size {size}: expected ({size},), got {result.shape}"
            
            # Assertion to verify the tensor is on the expected device (CPU by default here)
            assert result.device == torch.device('cpu'), "Device mismatch"
            
        except Exception as e:
            pytest.fail(
                f"torch.compile failed to handle dynamic size {size}. "
                f"This reproduces the regression reported in Issue 161372. Error: {e}"
            )

if __name__ == "__main__":
    test_compile_dynamic_hann_window()
    print("Test passed successfully.")