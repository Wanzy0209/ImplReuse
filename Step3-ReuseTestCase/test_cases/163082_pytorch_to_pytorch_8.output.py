import torch

# Ensure reproducibility
torch.manual_seed(1337)

# Define the compiled version of the similar API
@torch.compile()
def leaky_relu_compiled(input_tensor):
    return torch.nn.functional.leaky_relu(input_tensor)

# Define the eager version of the similar API
def leaky_relu_eager(input_tensor):
    return torch.nn.functional.leaky_relu(input_tensor)

# Check for CUDA availability
if torch.cuda.is_available():
    device = 'cuda'
    # Create a tensor with float32, including negative values to test the 'leaky' part
    # Using a structure similar to the original bug report's input
    input_tensor = torch.tensor([[3.799999, -1.5, 0.0]], device=device, dtype=torch.float32)
    
    print("Input vector:", [x.item() for x in input_tensor[0]])

    # Run compiled version
    res_compiled = leaky_relu_compiled(input_tensor)
    print("Output (compiled):", [x.item() for x in res_compiled[0]])

    # Run eager version
    res_eager = leaky_relu_eager(input_tensor)
    print("Output (eager):", [x.item() for x in res_eager[0]])

    # Assert that the results are close enough to match expected behavior
    # The original bug showed a deviation where norm > 1. Here we check for exactness
    # between compiled and eager paths to catch similar numerical discrepancies.
    assert torch.allclose(res_compiled, res_eager, rtol=1e-5, atol=1e-8), \
        f"Mismatch between compiled and eager results: {res_compiled} vs {res_eager}"
    
    print("Test passed: Compiled and eager outputs match.")
else:
    print("CUDA is not available. Skipping test.")