import torch

# Replacing torch.randint with torch.amax
# To generate tensor1 of shape (9, 3, 7) with dtype int16 and values in range [-100, 100]
# we create a larger random tensor, scale it, and use torch.amax to reduce the last dimension.
source1 = torch.rand(9, 3, 7, 5) * 200 - 100
tensor1 = torch.amax(source1, dim=-1).to(torch.int16)

# To generate tensor2 of shape (1, 6, 4, 8) with dtype bool
# we create a larger random tensor, use torch.amax, and threshold the result.
source2 = torch.rand(1, 6, 4, 8, 5)
tensor2 = (torch.amax(source2, dim=-1) > 0.5).to(torch.bool)

# Original test case logic to trigger the segmentation fault
input_data = [[[], 154691921484029491302139942063978250367, ()], {}, [tensor1, tensor2], {}]
r1 = torch.nn.MaxUnpool2d(*input_data[0], **input_data[1])
r2 = r1(*input_data[2], **input_data[3])