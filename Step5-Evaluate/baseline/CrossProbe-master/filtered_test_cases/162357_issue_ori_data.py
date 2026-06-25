import torch
from torch._inductor import compile_fx

def test_aoti_fx_add():
    def fn(x, y):
        return x + y
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    compiled_fn = compile_fx(fn, (x, y))
    result = compiled_fn(x, y)
    expected = fn(x, y)
    assert torch.allclose(result, expected)