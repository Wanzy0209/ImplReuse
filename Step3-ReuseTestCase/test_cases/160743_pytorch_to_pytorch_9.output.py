import torch
torch.manual_seed(0)

# Adapted test case for torch.nn.NLLLoss
# NLLLoss expects log-probabilities as input and class indices as target
batch_size = 4
num_classes = 6

# Input: (N, C) representing log-probabilities
x = torch.randn(batch_size, num_classes)
# Target: (N) representing class indices
target = torch.randint(0, num_classes, (batch_size,))

# Instantiate NLLLoss
# Using reduction='none' to output a tensor per sample, similar to the spatial output of AvgPool2d
model = torch.nn.NLLLoss(reduction='none')

out_cpu = model(x, target)
out_mps = model(x.to("mps"), target.to("mps"))

if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
    print("Output does not match!")
    print(out_cpu)
    print(out_mps)