import torch

# Enable recompilation logging as in the original bug report
torch._logging.set_logs(recompiles=True)

# Define a function using the similar API: torch.prod
def model_fn(x):
    return torch.prod(x)

# Compile the function (mimicking the compile_repeated_blocks call)
compiled_model = torch.compile(model_fn)

# Setup input (mimicking the pipeline inputs)
# Using CUDA if available to match the original context
device = "cuda" if torch.cuda.is_available() else "cpu"
input_data = torch.randn(2, 3, 4, device=device)

# Run the compiled model (mimicking the pipe(...) call)
# This will trigger the compilation and potentially show recompiles if the bug exists
result = compiled_model(input_data)

# Verify the output
expected = torch.prod(input_data)
assert torch.allclose(result, expected), "Output mismatch for torch.prod"

# Run a second time to check for recompilation behavior
input_data_2 = torch.randn(2, 3, 4, device=device)
result_2 = compiled_model(input_data_2)
assert torch.allclose(result_2, torch.prod(input_data_2))