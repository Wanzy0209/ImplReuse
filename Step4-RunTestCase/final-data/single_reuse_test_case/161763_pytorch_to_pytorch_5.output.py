import torch

def foo(x):
    # Adapted to use torch.any instead of neg/add
    c = torch.tensor(7, dtype=torch.uint8)
    return torch.any(x), torch.any(c)

torch.manual_seed(0)
# Create a random uint8 tensor with values 0 or 1 to test torch.any
x = torch.randint(0, 2, (2, 2), dtype=torch.uint8)
print(f"input: {x}")

cfoo = torch.compile(foo)
res = foo(x)
cres = cfoo(x)

print(f"res[0]: {res[0]}")
print(f"cres[0]: {cres[0]}")
assert res[0] == cres[0], f"Mismatch in result 0: {res[0]} != {cres[0]}"

print(f"res[1]: {res[1]}")
print(f"cres[1]: {cres[1]}")
assert res[1] == cres[1], f"Mismatch in result 1: {res[1]} != {cres[1]}"

print("Test passed.")