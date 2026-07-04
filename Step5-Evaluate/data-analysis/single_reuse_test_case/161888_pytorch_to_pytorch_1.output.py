import torch
print(torch.__version__)

# Fix 1: Use valid kernel_size (list of 3 ints)
# Fix 2: Use valid stride (int)
# Fix 3: Use valid padding (tuple of 3 ints)
# Original: [[], 154691921484029491302139942063978250367, ()]
# The empty list for kernel_size and the huge integer for stride were causing the TypeError.
input_args = [[2, 2, 2], 2, (0, 0, 0)]
input_kwargs = {}

# Fix 4: Use valid 5D input tensor for MaxUnpool3d
# Original shape: (9, 3, 7) -> 3D (Invalid)
# New shape: (1, 3, 9, 3, 7) -> 5D (Batch, Channel, Depth, Height, Width)
tensor1 = torch.randint(
    low=-100,
    high=100,
    size=(1, 3, 9, 3, 7),
    dtype=torch.int16
)

# Fix 5: Use valid indices tensor (must match input shape and be int64)
# Original shape: (1, 6, 4, 8) -> 4D (Invalid), dtype: bool (Invalid)
# New shape: (1, 3, 9, 3, 7) -> 5D, dtype: int64
tensor2 = torch.randint(
    low=0,
    high=2, 
    size=(1, 3, 9, 3, 7),
    dtype=torch.int64
)

input = [input_args, input_kwargs, [tensor1, tensor2], {}]

r1 = torch.nn.MaxUnpool3d(*input[0], **input[1])
r2 = r1(*input[2], **input[3])