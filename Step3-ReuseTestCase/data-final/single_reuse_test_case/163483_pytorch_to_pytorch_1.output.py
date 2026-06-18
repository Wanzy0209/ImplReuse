import torch

torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(rank)

# Create a tensor with channels_last memory format
x = torch.arange(0, 16, dtype=torch.int64).reshape(2, 2, 2, 2).cuda().to(memory_format=torch.channels_last)

# Prepare input list for reduce_scatter
# We populate the list with the same tensor 'x' for simplicity.
# input_list[i] will be reduced across ranks and scattered to rank i.
input_list = [x.clone() for _ in range(world_size)]

# Prepare output tensor with channels_last format
# This mimics the setup in the bug report where the output buffer is expected to maintain the format.
output = torch.zeros_like(x)

# Perform reduce_scatter
torch.distributed.reduce_scatter(output, input_list)

# Expected result: x * world_size (sum of 'x' from all ranks)
expected = x * world_size

# Check if the output matches the expected value
# If the memory ordering bug exists, this will likely fail or the data will be scrambled.
is_equal = torch.equal(output, expected)

print('rank_{}: {}\n x:\n{}\n output:\n{}\n expected:\n{}\n'.format(
    rank, is_equal, x.storage(), output.storage(), expected.storage()
))