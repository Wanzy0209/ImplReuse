import torch
import sys

print("PyTorch version:", torch.__version__)

# Setup tensors as in the original bug report
tensor1 = torch.randint(
    low=-100,
    high=100,
    size=(9, 3, 7),
    dtype=torch.int16
)
tensor2 = torch.randint(
    low=0,
    high=2,
    size=(1, 6, 4, 8),
    dtype=torch.bool
)

# Reuse the malformed input structure from the original bug to test NLLLoss
# The error "cannot assign 'list' object to buffer 'weight'" indicates that the first argument
# in input_data[0] (which is []) is being passed as the 'weight' parameter to NLLLoss.
# NLLLoss expects 'weight' to be a Tensor or None, not a list.
# To fix this while keeping the test structure, we change the list to None.
# We also fix the other arguments to be valid types to allow the test to proceed to the forward pass
# or fail on the intended logic rather than argument validation.
# input[0]: args for __init__ -> [weight, size_average, ignore_index]
# input[1]: kwargs for __init__ -> {}
# input[2]: args for forward -> [tensor1, tensor2]
# input[3]: kwargs for forward -> {}

# Fixed arguments: weight=None (was []), size_average=None (was huge int), ignore_index=-100 (was ())
input_data = [[None, None, -100], {}, [tensor1, tensor2], {}]

try:
    # Test torch.nn.NLLLoss with the inputs
    r1 = torch.nn.NLLLoss(*input_data[0], **input_data[1])
    r2 = r1(*input_data[2], **input_data[3])
    print("Test completed. Result:", r2)
except Exception as e:
    # Catching standard exceptions to differentiate from a Segmentation Fault
    print(f"Caught Exception: {type(e).__name__}: {e}")
    sys.exit(1)