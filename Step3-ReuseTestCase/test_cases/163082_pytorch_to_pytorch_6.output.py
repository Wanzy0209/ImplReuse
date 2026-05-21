import torch

torch.manual_seed(1337)

@torch.compile()
def log_softmax_compiled(x):
    return torch.nn.LogSoftmax(dim=1)(x)

def log_softmax_no_compile(x):
    return torch.nn.LogSoftmax(dim=1)(x)

device='cuda'
# Using a similar tensor structure to the original bug report to test numerical stability
c=torch.tensor([[3.799999 ,0.0, 0.0]],device=device,dtype=torch.float32)

print("Input vector",[x.item() for x in c[0] ])

# Run compiled version
xyz_compiled=log_softmax_compiled(c)
print("LogSoftmax vector (compile):",[x.item() for x in xyz_compiled[0] ])

# Run non-compiled version
xyz_no_compile=log_softmax_no_compile(c)
print("LogSoftmax vector (without compile):",[x.item() for x in xyz_no_compile[0] ])

# Verify the mathematical property: sum(exp(x)) should be 1
sum_exp_compiled = torch.exp(xyz_compiled).sum().item()
sum_exp_no_compile = torch.exp(xyz_no_compile).sum().item()

print(f"Sum of exps (compile): {sum_exp_compiled}")
print(f"Sum of exps (without compile): {sum_exp_no_compile}")

# Assert that the compiled version maintains the same mathematical properties as the non-compiled version
# and that the sum of exponentials is close to 1.0
assert torch.allclose(xyz_compiled, xyz_no_compile, atol=1e-6), "Compiled and non-compiled outputs differ significantly"
assert torch.allclose(torch.exp(xyz_compiled).sum(), torch.tensor(1.0), atol=1e-6), "Sum of exps is not 1"