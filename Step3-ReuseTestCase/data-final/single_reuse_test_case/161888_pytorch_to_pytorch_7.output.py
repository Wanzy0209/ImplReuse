import torch
print(torch.__version__)

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
# Adapted input structure to test torch.nn.CosineEmbeddingLoss
test_args = [[[], 154691921484029491302139942063978250367, ()],{},[tensor1,tensor2],{}]
r1 = torch.nn.CosineEmbeddingLoss(*test_args[0],**test_args[1])
r2 = r1(*test_args[2],**test_args[3])