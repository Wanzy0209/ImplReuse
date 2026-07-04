import torch

# Adapt the test case to use torch.dist
# We create tensors on CUDA similar to the original failing case (torch.ones(1, device="cuda"))
# to check if the memory allocation issue affects this API as well.
input1 = torch.ones(1, device="cuda")
input2 = torch.zeros(1, device="cuda")

# Call the similar API torch.dist
# This involves operations (subtraction and norm) that allocate memory on the device.
result = torch.dist(input1, input2)

# Verify the result (distance between 1 and 0 is 1.0)
print(result.item())
assert result.item() == 1.0