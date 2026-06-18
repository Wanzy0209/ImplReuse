import torch

print(f"PyTorch version: {torch.__version__}")

# Define tensors with specific dtypes and shapes from the original bug report
# tensor1 is 3D (valid for 1D input), tensor2 is 4D (invalid for 1D indices)
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

# Prepare the input arguments structure
# input[0]: Init args (kernel_size, stride, padding) -> Malformed: [], huge_int, ()
# input[1]: Init kwargs -> {}
# input[2]: Forward args (input, indices) -> Mismatched tensors
# input[3]: Forward kwargs -> {}
input_data = [[[], 154691921484029491302139942063978250367, ()], {}, [tensor1, tensor2], {}]

# Instantiate MaxUnpool1d with malformed arguments
r1 = torch.nn.MaxUnpool1d(*input_data[0], **input_data[1])

# Call forward with mismatched tensor types/shapes
r2 = r1(*input_data[2], **input_data[3])