import torch
A = torch.rand(5, 5, device="cuda" if torch.cuda.is_available() else "cpu")
def f(A, count):
    Q, R = torch.linalg.qr(A)
    rhs = torch.ones(Q.shape[0], 1, device=A.device)
    a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
    if a.stride() == a.clone(memory_format=torch.preserve_format).stride():
        return count + 1
    return count

res1 = f(A, torch.zeros(1))
print(res1)

res2 = torch.compile(f)(A, torch.zeros(1))
print(res2)