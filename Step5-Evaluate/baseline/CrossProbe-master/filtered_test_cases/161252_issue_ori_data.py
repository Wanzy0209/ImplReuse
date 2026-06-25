import torch
m = torch.nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)
print(output)