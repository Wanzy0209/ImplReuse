import torch
import numpy as np

torch._dynamo.config.capture_scalar_outputs = True

def foo(x):
    t = torch.tan(x)
    e = t.expand(31, 51, 1)
    
    # Adapted to use torch.any instead of torch.mean
    # Checking if any element in e is greater than 0.5
    any_val = torch.any(e > 0.5)
    
    # Using the result of torch.any for control flow
    if any_val.item():
        out1 = torch.sub(e, e * 0.5)
    else:
        out1 = torch.add(e, e * 0.5)
        
    return torch.sin(out1)

np.random.seed(0)
x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)

cfoo = torch.compile(foo)
eager_res = foo(torch.from_numpy(x))
compile_res = cfoo(torch.from_numpy(x))

torch.testing.assert_close(eager_res, compile_res)