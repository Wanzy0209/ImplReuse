import torch
from torch.testing._internal.common_methods_invocations import op_db

# Find the op_db entry for cholesky_solve
op_dict = next((op for op in op_db if op.name == "cholesky_solve"), None)

if op_dict is None:
    raise RuntimeError("Could not find cholesky_solve in op_db")

def fn(input1, input2):
    return torch.cholesky_solve(input1, input2)

# Sample inputs using "cuda" and float32 to match the original bug context
inputs = list(op_dict.sample_inputs("cuda", torch.float32, requires_grad=False))

for sample in inputs:
    # Unpack arguments: sample.input is the first arg, sample.args[0] is the second
    # cholesky_solve takes (input, input2, upper=False). We pass the two tensors.
    arg1 = sample.input
    arg2 = sample.args[0]

    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
    res1 = fn(arg1, arg2)
    res2 = compiled(arg1, arg2)
    torch.testing.assert_close(res1, res2)