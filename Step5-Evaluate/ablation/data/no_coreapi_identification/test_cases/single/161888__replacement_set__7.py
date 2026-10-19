import torch
print(torch.__version__)

# Replace torch.randint with torch.argmin
# To generate tensors of specific shapes using argmin, we create a source tensor and reduce a dimension.
src1 = torch.randn(9, 3, 7, 1)
tensor1 = torch.argmin(src1, dim=-1)

src2 = torch.randn(1, 6, 4, 8, 1)
tensor2 = torch.argmin(src2, dim=-1)

input = [[[], 154691921484029491302139942063978250367, ()],{},[tensor1,tensor2],{}]
r1 = torch.nn.MaxUnpool2d(*input[0],**input[1])
r2 = r1(*input[2],**input[3])