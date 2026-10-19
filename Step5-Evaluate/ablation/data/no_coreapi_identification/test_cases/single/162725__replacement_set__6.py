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
    # Replacement: torch.compile -> torch.any
    # We assign the function torch.any to 'compiled' to replace the compilation step
    compiled = torch.any
    res1 = fn(x,w)
    # Adapt the call to match torch.any signature (takes input tensor)
    res2 = compiled(x)
    # Replacement: torch.testing.assert_close -> torch.nn.utils.rnn.pack_sequence
    torch.nn.utils.rnn.pack_sequence([res1, res2])