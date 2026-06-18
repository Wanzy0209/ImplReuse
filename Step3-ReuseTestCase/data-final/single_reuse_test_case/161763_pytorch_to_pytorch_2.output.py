import torch


def foo(x):
    # Adapted to use torch.prod with uint8 tensor
    c = torch.tensor([2, 3, 4], dtype=torch.uint8)
    # Compute product of c and add to x
    return torch.prod(c), torch.prod(c) + x
    

torch.manual_seed(0)
x = torch.randn(2, 2, dtype=torch.float32)
print(f"input: {x}")
'''
input: tensor([[ 1.5410, -0.2934],
        [-2.1788,  0.5684]])
'''
cfoo = torch.compile(foo)
res = foo(x)
cres = cfoo(x)

print(f"res[0]: {res[0]}")
print(f"cres[0]: {cres[0]}")
assert torch.equal(res[0], cres[0]), "Mismatch in product result"
'''
res[0]: 24
cres[0]: 24
'''

print(f"res[1]: {res[1]}")
print(f"cres[1]: {cres[1]}")
assert torch.allclose(res[1], cres[1]), "Mismatch in addition result"
'''
res[1]: tensor([[25.5410, 23.7066],
        [21.8212, 24.5684]])
cres[1]: tensor([[25.5410, 23.7066],
        [21.8212, 24.5684]])
'''