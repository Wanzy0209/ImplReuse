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

# Adjusted input to match BatchNorm1d API:
# - input[0]: BatchNorm1d expects num_features (int). Using 3 based on tensor1's channel dim.
# - input[2]: BatchNorm1d.forward expects 1 input tensor. Using tensor1.
input = [[3], {}, [tensor1], {}]

# Replacing torch.nn.MaxUnpool2d with torch.nn.BatchNorm1d
# to test if the similar API handles the malformed inputs robustly.
r1 = torch.nn.BatchNorm1d(*input[0],**input[1])
r2 = r1(*input[2],**input[3])