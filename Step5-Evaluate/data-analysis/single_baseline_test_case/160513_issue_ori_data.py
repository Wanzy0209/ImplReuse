# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
print(torch.version.__version__)

fargs = {'dtype': torch.float32, 'device': 'cuda'}
A = torch.rand((5, 3, 100), **fargs)

def test(x):
    a = 0.5 * x[0]
    p, t = torch.autograd.forward_ad.unpack_dual(a)
    print(f"a dtype : {p.dtype}, a.tangent dtype : {t.dtype}")
    v = torch.stack([a, *x[1:]])
    p, t = torch.autograd.forward_ad.unpack_dual(v)
    print(f"v dtype : {p.dtype}, v.tangent dtype : {t.dtype}")
    return torch.tensordot(v, A, dims=1)

torch.func.jacfwd(test)(torch.rand(5, **fargs))