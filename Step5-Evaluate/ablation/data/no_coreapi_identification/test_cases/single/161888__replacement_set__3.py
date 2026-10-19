import torch

# Replacing torch.randint with torch.fmin
# Original: torch.randint(low=-100, high=100, size=(9, 3, 7), dtype=torch.int16)
tensor1 = torch.fmin(
    torch.full((9, 3, 7), -100, dtype=torch.int16),
    torch.full((9, 3, 7), 100, dtype=torch.int16)
)

# Original: torch.randint(low=0, high=2, size=(1, 6, 4, 8), dtype=torch.bool)
tensor2 = torch.fmin(
    torch.zeros((1, 6, 4, 8), dtype=torch.bool),
    torch.ones((1, 6, 4, 8), dtype=torch.bool)
)

input = [[[], 154691921484029491302139942063978250367, ()],{},[tensor1,tensor2],{}]
r1 = torch.nn.MaxUnpool2d(*input[0],**input[1])
r2 = r1(*input[2],**input[3])