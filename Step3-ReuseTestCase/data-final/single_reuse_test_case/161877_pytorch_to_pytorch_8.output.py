import torch

# Setup input data similar to the original bug report
input_data = torch.randn(1, 16, 100)

# The extreme value that caused the crash in Conv1d
extreme_value = 9223372036854775803

# Test torch.randint_like with the extreme value
# Note: randint_like requires explicit dtype when input is float
try:
    output = torch.randint_like(input_data, extreme_value, dtype=torch.int64)
    # If we reach here, the API handled the extreme value without crashing
    assert output.shape == input_data.shape
    print("Test completed successfully.")
except RuntimeError as e:
    # Catching potential runtime errors (e.g. value out of range for dtype)
    print(f"RuntimeError caught: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")