import torch

# Adapted test case for torch.std_mean based on Issue 161877
# The original issue involved passing an extremely large integer (padding) to Conv1d,
# causing a crash (realloc(): invalid pointer).
# Here we test if passing a similarly large integer to the 'dim' parameter of torch.std_mean
# causes a crash or similar instability.

input_data = torch.randn(1, 16, 100)
extreme_value = 9223372036854775803

try:
    # Pass the extreme value to the 'dim' parameter
    output = torch.std_mean(input_data, dim=extreme_value)
    print("Test passed. Output:", output)
except RuntimeError as e:
    # Expected behavior for valid bounds checking, but we verify it doesn't crash the process
    print(f"Caught RuntimeError (expected bounds check): {e}")
except Exception as e:
    print(f"Caught unexpected exception: {type(e).__name__}: {e}")