import torch
import torch.nn.functional as F

# Adapted test case for torch.nn.functional.embedding_bag
# based on the share_memory_() segfault issue with NestedTensor.

# Setup inputs mimicking the jagged structure from the original bug report
# Original: a = torch.randn(3), b = torch.randn(5) -> offsets [0, 3, 8]
input_indices = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4], dtype=torch.long)
weight = torch.randn(10, 3)
offsets = torch.tensor([0, 3, 8], dtype=torch.long)

# Call the similar API
output = F.embedding_bag(input_indices, weight, offsets=offsets)

# Perform the action that caused the crash in the original API
# We expect this to succeed without a segmentation fault
output.share_memory_()

# Verify the tensor is actually in shared memory
assert output.is_shared(), "Output tensor should be in shared memory"