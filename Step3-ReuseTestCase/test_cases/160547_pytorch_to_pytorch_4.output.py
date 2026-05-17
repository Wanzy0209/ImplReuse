import torch
import torch.nn as nn
from collections import namedtuple
from torch.utils.checkpoint import checkpoint_sequential

def test_checkpoint_sequential_namedtuple():
    Point = namedtuple('Point', 'x y')
    
    # Adapted module: checkpoint_sequential passes input directly to the function,
    # it does not unpack args like export does. So the module must accept the tuple.
    class M(nn.Module):
        def forward(self, p):
            return p.x + p.y 
    
    inp = Point(torch.ones(3), torch.ones(3))
    modules = [M()]
    
    # Expected output
    expected = M()(inp)
    
    print("Testing torch.utils.checkpoint.checkpoint_sequential with NamedTuple input")
    
    # Test with use_reentrant=False (analogous to strict=False in terms of being a mode switch)
    try:
        out = checkpoint_sequential(modules, 1, inp, use_reentrant=False)
        assert torch.allclose(out, expected)
        print("use_reentrant=False succeeded")
    except Exception as e:
        print(f"use_reentrant=False failed: {e}")

    # Test with use_reentrant=True (analogous to strict=True)
    try:
        out = checkpoint_sequential(modules, 1, inp, use_reentrant=True)
        assert torch.allclose(out, expected)
        print("use_reentrant=True succeeded")
    except Exception as e:
        print(f"use_reentrant=True failed: {e}")

if __name__ == "__main__":
    test_checkpoint_sequential_namedtuple()