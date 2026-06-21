import torch

# Define a simple wrapper to mimic the structure of sample inputs
# since we cannot import op_db due to missing internal dependencies (expecttest).
class SampleInput:
    def __init__(self, input, args):
        self.input = input
        self.args = args

# Adapt the function to match the batch_norm API.
def fn(*args):
    return torch.nn.functional.batch_norm(*args)

# Manually create sample inputs for CUDA and float32
# This replaces the logic that relied on torch.testing._internal
inputs = []

# Create base tensors
# Shape: (Batch, Channels, Height, Width) -> (2, 3, 4, 4)
input_tensor = torch.randn(2, 3, 4, 4, device="cuda", dtype=torch.float32, requires_grad=False)
running_mean = torch.randn(3, device="cuda", dtype=torch.float32)
# running_var must be positive, so we take absolute value and add a small epsilon
running_var = torch.randn(3, device="cuda", dtype=torch.float32).abs() + 0.1
weight = torch.randn(3, device="cuda", dtype=torch.float32)
bias = torch.randn(3, device="cuda", dtype=torch.float32)

# Case 1: Training=False (Inference mode)
inputs.append(SampleInput(
    input_tensor,
    (running_mean, running_var, weight, bias, False)
))

# Case 2: Training=True (Training mode)
# Note: running_mean/var are updated in-place in training mode, so we clone them for the second test
inputs.append(SampleInput(
    input_tensor,
    (running_mean.clone(), running_var.clone(), weight, bias, True)
))

for sample in inputs:
    # Reconstruct the full argument list: input tensor followed by other args
    eager_args = (sample.input,) + sample.args

    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")

    # Run eager and compiled versions
    res1 = fn(*eager_args)
    res2 = compiled(*eager_args)

    # Assert consistency
    torch.testing.assert_close(res1, res2)