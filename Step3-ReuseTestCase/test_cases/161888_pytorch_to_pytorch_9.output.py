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
# input[0]: args for __init__ -> [], huge_int, ()
# input[1]: kwargs for __init__ -> {}
# input[2]: args for forward -> [tensor1, tensor2]
# input[3]: kwargs for forward -> {}
input_data = [[[], 154691921484029491302139942063978250367, ()], {}, [tensor1, tensor2], {}]

try:
    # Test torch.nn.NLLLoss with the malformed inputs
    r1 = torch.nn.NLLLoss(*input_data[0], **input_data[1])
    r2 = r1(*input_data[2], **input_data[3])
    print("Test completed. Result:", r2)
except Exception as e:
    # Catching standard exceptions to differentiate from a Segmentation Fault
    print(f"Caught Exception: {type(e).__name__}: {e}")
    sys.exit(1)