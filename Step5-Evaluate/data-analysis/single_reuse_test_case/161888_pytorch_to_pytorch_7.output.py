import torch
print(torch.__version__)

# To fix the error, we need compatible input tensors and a target tensor.
# CosineEmbeddingLoss requires input1 and input2 to have the same shape (N, D)
# and target to have shape (N,) with values 1 or -1.

# Using compatible shapes (e.g., batch_size=9, dim=7)
tensor1 = torch.randint(
    low=-100,
    high=100,
    size=(9, 7),
    dtype=torch.int16
)
tensor2 = torch.randint(
    low=0,
    high=2,
    size=(9, 7),
    dtype=torch.bool
)

# Generate target tensor (1 or -1)
tensor3 = torch.randint(0, 2, (9,))
tensor3[tensor3 == 0] = -1

# Fix constructor arguments. 
# Original: [[[], 154691921484029491302139942063978250367, ()], ...]
# Fixed: [[0.0], ...] (margin=0.0)
test_args = [[0.0], {}, [tensor1, tensor2, tensor3], {}]

r1 = torch.nn.CosineEmbeddingLoss(*test_args[0], **test_args[1])
r2 = r1(*test_args[2], **test_args[3])
print(r2)