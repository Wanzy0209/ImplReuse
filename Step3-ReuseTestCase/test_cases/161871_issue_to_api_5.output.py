import torch
import torch.nn.functional as F

# Reproduce the tensor creation logic from the bug report
# The original issue involved an int64 tensor created with randint
tensor = torch.randint(low=0, high=10, size=(5,), dtype=torch.int64)

# Adapt the input structure for the similar API (sigmoid)
# The original bug used [[tensor, 1024], {}] for mvlgamma_
# Sigmoid only takes the input tensor, so we adjust the arguments list accordingly
input_args = [[tensor], {}]

# Call the similar API using the unpacking pattern from the bug report
# This verifies that the unpacking mechanism and int64 input handling work correctly
# for torch.nn.functional.sigmoid, contrasting with the crash in the original issue.
result = F.sigmoid(*input_args[0], **input_args[1])

# Assertion: Verify the operation completed successfully and returned the expected type
assert result is not None
assert result.dtype == torch.float32
print("Test passed.")