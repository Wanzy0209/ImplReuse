# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


def foo(x):
    c = torch.tensor(7, dtype=torch.uint8)
    return c+x, torch.neg(c), torch.neg(c)+x
    

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
'''
res[0]: tensor([[8.5410, 6.7066],
        [4.8212, 7.5684]])
cres[0]: tensor([[8.5410, 6.7066],
        [4.8212, 7.5684]])
'''
print(f"res[1]: {res[1]}")
print(f"cres[1]: {cres[1]}")
'''
res[1]: 249
cres[1]: 249
'''
print(f"res[2]: {res[2]}")
print(f"cres[2]: {cres[2]}")
'''
res[2]: tensor([[250.5410, 248.7066],
        [246.8212, 249.5684]])
cres[2]: tensor([[-5.4590, -7.2934],
        [-9.1788, -6.4316]])
'''