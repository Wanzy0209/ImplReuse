# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import numpy as np

torch._dynamo.config.capture_scalar_outputs = True

def foo(x):
    t = torch.tan(x)
    e = t.expand(31, 51, 1)
    mean_val = torch.mean(e)
    if mean_val.item() > 0.5:
        out1 = torch.sub(e, e * 0.5)
    else:
        out1 = torch.add(e, e * 0.5)
    # print("break")  # no error occurs if uncomment this line
    return torch.sin(out1)


np.random.seed(0)
x = np.random.uniform(0, 10, size=(31, 51, 1)).astype(np.float16)

cfoo = torch.compile(foo)
eager_res = foo(torch.from_numpy(x))
compile_res = cfoo(torch.from_numpy(x))

torch.testing.assert_close(eager_res, compile_res)