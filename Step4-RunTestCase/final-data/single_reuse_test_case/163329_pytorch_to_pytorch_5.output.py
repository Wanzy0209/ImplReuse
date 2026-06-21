import torch

# Enable recompilation logging as seen in the original bug report
# Handle cases where torch._logging is not available in the current version
try:
    torch._logging.set_logs(recompiles=True)
except AttributeError:
    pass  # Ignore if the logging API is not available

def test_torch_any_compile():
    # Define a function using the similar API: torch.any
    def any_fn(x):
        return torch.any(x > 0)

    # Compile the function
    compiled_any_fn = torch.compile(any_fn)

    # Test with a random tensor
    x = torch.randn(10, 10)
    
    # Run compiled version
    compiled_result = compiled_any_fn(x)
    
    # Run eager version for verification
    eager_result = any_fn(x)
    
    # Assert correctness
    assert compiled_result == eager_result, f"Mismatch: compiled={compiled_result}, eager={eager_result}"
    
    # Test with a different shape to check for potential recompilation issues
    y = torch.randn(5, 5)
    compiled_result_y = compiled_any_fn(y)
    eager_result_y = any_fn(y)
    assert compiled_result_y == eager_result_y, f"Mismatch on different shape: compiled={compiled_result_y}, eager={eager_result_y}"

if __name__ == "__main__":
    test_torch_any_compile()
    print("Test passed.")