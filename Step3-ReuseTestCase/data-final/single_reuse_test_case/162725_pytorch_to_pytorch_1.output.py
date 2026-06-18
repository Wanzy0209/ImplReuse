import torch
from torch.testing._internal.common_methods_invocations import op_db

def fn(x, w):
    # Adapted to use the similar API: conv_transpose1d
    result = torch.nn.functional.conv_transpose1d(input=x, weight=w)
    return result

# Find the op_db entry for conv_transpose1d
op_dict = next((op for op in op_db if "conv_transpose1d" in op.name), None)

if op_dict:
    inputs = list(op_dict.sample_inputs("cuda", torch.float32, requires_grad=False))

    for sample in inputs:
        eager_args = sample.input, *sample.args
        (x, w, b) = eager_args
        compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
        res1 = fn(x, w)
        res2 = compiled(x, w)
        torch.testing.assert_close(res1, res2)
else:
    print("op_db entry for conv_transpose1d not found.")