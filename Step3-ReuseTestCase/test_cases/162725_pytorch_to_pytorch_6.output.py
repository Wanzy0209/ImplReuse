import torch
from torch.testing._internal.common_methods_invocations import op_db

def fn(input, mat, vec, beta=1, alpha=1):
    result = torch.addmv(input, mat, vec, beta=beta, alpha=alpha)
    return result

# Find the op_db entry for torch.addmv
op_dict = None
for op in op_db:
    if op.name == "addmv":
        op_dict = op
        break

if op_dict:
    # Sample inputs using the same configuration as the original bug report
    inputs = list(op_dict.sample_inputs("cuda", torch.float32, requires_grad=False))

    for sample in inputs:
        # Prepare arguments: sample.input is the first arg, sample.args are the rest
        # addmv signature: (input, mat, vec, beta=1, alpha=1)
        args = (sample.input, *sample.args)
        
        compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
        
        res1 = fn(*args)
        res2 = compiled(*args)
        
        torch.testing.assert_close(res1, res2)
else:
    print("torch.addmv not found in op_db")