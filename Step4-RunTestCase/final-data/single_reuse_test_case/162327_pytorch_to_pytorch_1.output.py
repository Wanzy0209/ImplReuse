import torch

print(torch.__version__, flush=True)

# Replicating the input structure from the original bug report
# to test for similar buffer overflow issues in max_unpool3d
input = [
    [
        torch.empty((5, 7, 4, 3, 7, 6), dtype=torch.int8),
        torch.empty((4, 9, 2), dtype=torch.int32),
        (1, 1, 1),  # Fixed: kernel_size must be a tuple of 3 ints
        (1, 1, 1)   # Fixed: stride must be a tuple of 3 ints
    ],
    {},
    [],
    {}
]

# Call the similar API
torch.nn.functional.max_unpool3d(*input[0], **input[1])