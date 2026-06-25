import torch
from torch.fx import symbolic_trace, Interpreter

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 1

mod = MyModule()
gm = symbolic_trace(mod)
interp = Interpreter(gm)
# This should fail but silently ignores extra args
result = interp.boxed_run([torch.tensor(1.0), torch.tensor(2.0)])
print(result)