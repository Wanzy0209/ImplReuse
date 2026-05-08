import torch
from torch.testing._internal.common_methods_invocations import op_db

def fn(x,w):
    result = torch.nn.functional.conv_transpose3d(input=x, weight=w)
    return result


op_dict = op_db[192]  # nn.functional.conv_transpose3d
inputs = list(op_dict.sample_inputs("cuda", torch.float32, requires_grad=False))

for sample in inputs:
    eager_args = sample.input, *sample.args
    (x,w,b) = eager_args
    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
    res1 = fn(x,w)
    res2 = compiled(x,w)
    torch.testing.assert_close(res1, res2)