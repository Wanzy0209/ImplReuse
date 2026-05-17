import torch
import torch.distributed as dist
import tempfile
import os

# Initialize the process group for the distributed operation
# Note: This requires the 'gloo' backend. For a single-process test, we use a temporary file store.
if not dist.is_initialized():
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{temp_file.name}",
        rank=0,
        world_size=1
    )

# Create a tensor
tensor = torch.arange(4, dtype=torch.int64)

# Call the similar API: torch.distributed.all_to_all_single
# Since world_size is 1, the output should be identical to the input
output = tensor.all_to_all_single(
    output_split_sizes=None,
    input_split_sizes=None,
    group=dist.group.WORLD
)

# Verify the result
assert torch.equal(tensor, output)