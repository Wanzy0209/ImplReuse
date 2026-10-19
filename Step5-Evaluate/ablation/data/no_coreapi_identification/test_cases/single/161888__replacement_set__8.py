import torch

# Replacing torch.randint with torch.argmax
# To generate tensor1 of shape (9, 3, 7), we create a dummy tensor with an extra dimension and reduce it.
dummy_input1 = torch.randn(9, 3, 7, 10)
tensor1 = torch.argmax(dummy_input1, dim=-1).to(torch.int16)

# To generate tensor2 of shape (1, 6, 4, 8), we create a dummy tensor with an extra dimension and reduce it.
dummy_input2 = torch.randn(1, 6, 4, 8, 10)
tensor2 = torch.argmax(dummy_input2, dim=-1).to(torch.bool)

# The input structure designed to trigger the segmentation fault
input = [[[], 154691921484029491302139942063978250367, ()],{},[tensor1,tensor2],{}]

# This initialization is expected to cause a segmentation fault due to the invalid large integer argument
r1 = torch.nn.MaxUnpool2d(*input[0],**input[1])

# Forward call (may not be reached if the constructor crashes)
r2 = r1(*input[2],**input[3])