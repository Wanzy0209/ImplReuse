@torch.compile(fullgraph=False, backend="eager")
def func(a):
    u0, u1 = a.tolist()
    return a*u0*u1
func(torch.tensor([1,2]))