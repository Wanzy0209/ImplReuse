import torch
torch.manual_seed(0)

# Adapted test case for torch.nn.CrossEntropyLoss
# Original test used AvgPool2d with specific parameters. 
# Here we test CrossEntropyLoss with weights and 'none' reduction to verify tensor-level consistency.

# Input: Batch size 4, 6 classes (reusing dimensions from original input where applicable)
x = torch.randn(4, 6)
# Target: Class indices for the batch
target = torch.randint(0, 6, (4,))

# Define the loss function with specific parameters
# Using weight and reduction='none' to ensure the output is a tensor for comparison
weight = torch.rand(6)
model = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

out_cpu = model(x, target)
out_mps = model(x.to("mps"), target.to("mps"))

if not torch.allclose(out_cpu, out_mps.cpu(), atol=1e-2, rtol=1e-2):
    print("Output does not match!")
    print(out_cpu)
    print(out_mps.cpu())
else:
    print("Test passed!")