import torch

# Ensure reproducibility
torch.manual_seed(1337)

# Define the compiled function using the similar API (torch.nn.functional.dropout3d)
@torch.compile()
def dropout_compiled(input_tensor):
    return torch.nn.functional.dropout3d(input_tensor, p=0.5, training=True)

# Define the eager function
def dropout_eager(input_tensor):
    return torch.nn.functional.dropout3d(input_tensor, p=0.5, training=True)

# Setup device and input
# Note: The original bug was specific to CUDA.
# dropout3d expects 3D, 4D, or 5D input (N, C, D, H, W) or (C, D, H, W).
# We reshape the original input [[3.799999, 0.0, 0.0]] to (1, 1, 3, 1, 1) to fit the API requirements.
device = 'cuda'
c = torch.tensor([[[[[3.799999]], [[0.0]], [[0.0]]]]], device=device, dtype=torch.float32)

print("Input vector:", [x.item() for x in c.flatten()])

# Run compiled version
# We set a specific seed here because dropout is stochastic. 
# If the compiled implementation is correct, it should match the eager implementation given the same seed.
torch.manual_seed(42)
out_compiled = dropout_compiled(c)
print("Output (compile):", [x.item() for x in out_compiled.flatten()])

# Run eager version
torch.manual_seed(42)
out_eager = dropout_eager(c)
print("Output (without compile):", [x.item() for x in out_eager.flatten()])

# Assertion to verify correctness
assert torch.equal(out_compiled, out_eager), "Outputs differ between compiled and eager mode"