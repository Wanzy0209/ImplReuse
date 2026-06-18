import torch
from torch.testing._internal.common_methods_invocations import op_db

def fn(a, b):
    return torch.floor_divide(a, b)

# Find the operator in the database corresponding to floor_divide
# Note: We search by name to ensure we get the correct op_db entry
op_dict = None
for op in op_db:
    if op.name == "floor_divide":
        op_dict = op
        break

if op_dict:
    inputs = list(op_dict.sample_inputs("cuda", torch.float32, requires_grad=False))

    for sample in inputs:
        # floor_divide takes two arguments: input and other
        # sample.input is the first argument, sample.args contains the rest
        args = (sample.input, *sample.args)
        
        if len(args) >= 2:
            a, b = args[0], args[1]
            
            compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
            res1 = fn(a, b)
            res2 = compiled(a, b)
            torch.testing.assert_close(res1, res2)
else:
    print("floor_divide not found in op_db")