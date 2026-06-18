import torch

MAX = 3
BATCH = 37


def func(x, idxs):
    return x.square() * torch.nn.functional.one_hot(idxs, MAX)


def jacfunc(x, idxs):
    return torch.func.jacfwd(func, argnums=(0,))(x, idxs)


idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
x = torch.rand((BATCH, MAX), dtype=torch.float64)

# works
out = jacfunc(x, idxs)

# fails
jacfunc = torch.compile(jacfunc, dynamic=True)
out = jacfunc(x, idxs)