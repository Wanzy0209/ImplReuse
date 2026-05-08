import torch
import torch.nn as nn
from torch.func import functional_call, jvp

class Linearized(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base.eval()
        # Frozen reference params and trainable params (dict of nn.Parameter)
        p0 = dict(self.base.named_parameters(remove_duplicate=False))
        self.params0 = {k: nn.Parameter(v.detach().clone(), requires_grad=False) for k, v in p0.items()}
        self.params  = {k: nn.Parameter(v.detach().clone())                      for k, v in p0.items()}

    def f(self, params, x):
        return functional_call(self.base, params, (x,))

    def forward(self, x):
        dparams = {k: self.params[k] - self.params0[k] for k in self.params0}
        (_, dp) = jvp(lambda P: self.f(P, x), (self.params0,), (dparams,))
        return self.f(self.params0, x) + dp

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = Linearized(nn.Linear(10, 5))
        self.post = nn.Linear(5, 5)

    def forward(self, x, substitution_tensor):
        x = self.lin(x)
        x = self.post(x)
        # simple post loop that writes Longs
        out = torch.zeros_like(x).long()
        B, D = out.shape
        if substitution_tensor.shape[0] != D:
            raise RuntimeError("shape mismatch")
        for i in range(B):
            out[i, :] = substitution_tensor
        return out

def get_inputs(B=32):
    x = torch.randn(B, 10)
    sub = torch.randint(0, 10, (5,), dtype=torch.long)
    return (x, sub)

if __name__ == "__main__":
    m = Model().eval()
    # Eager works:
    _ = m(*get_inputs())
    # Export fails:
    ep = torch.export.export(m, get_inputs())  # AssertionError on 2.8.0