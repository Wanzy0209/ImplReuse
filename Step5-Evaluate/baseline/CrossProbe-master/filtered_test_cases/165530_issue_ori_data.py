import torch
import torch.nn.functional as F
x = torch.randn(1, 1, 64, 64, requires_grad=True)
def f(x): return F.scaled_dot_product_attention(x, x, x)
try:
    torch.func.jvp(f, (x,), (torch.ones_like(x),))
except NotImplementedError as e:
    print(e)