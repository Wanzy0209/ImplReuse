import torch

# Define the function to test
def fn(input1, input2):
    return torch.cholesky_solve(input1, input2)

# Helper class to mimic the structure of op_db sample inputs
# This allows us to keep the loop logic exactly the same as the original
class SampleInput:
    def __init__(self, input, args):
        self.input = input
        self.args = args

# Manually generate sample inputs for cholesky_solve
# cholesky_solve(input, input2) solves AX = input for X, where input2 is the Cholesky factor of A.
# input2 must be square (n x n), input must be (n x m).
inputs = []

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float32

    # Case 1: 3x3 matrix, 3x1 RHS
    n, m = 3, 1
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.T + torch.eye(n, device=device) # Make symmetric positive definite
    L = torch.cholesky(A) # input2
    B = torch.randn(n, m, dtype=dtype, device=device) # input1
    inputs.append(SampleInput(B, (L,)))

    # Case 2: 5x2 RHS
    n, m = 5, 2
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.T + torch.eye(n, device=device)
    L = torch.cholesky(A)
    B = torch.randn(n, m, dtype=dtype, device=device)
    inputs.append(SampleInput(B, (L,)))

    # Case 3: 4x4 matrix, 4x4 RHS
    n, m = 4, 4
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.T + torch.eye(n, device=device)
    L = torch.cholesky(A)
    B = torch.randn(n, m, dtype=dtype, device=device)
    inputs.append(SampleInput(B, (L,)))

    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")

    for sample in inputs:
        # Unpack arguments: sample.input is the first arg, sample.args[0] is the second
        # cholesky_solve takes (input, input2, upper=False). We pass the two tensors.
        arg1 = sample.input
        arg2 = sample.args[0]

        res1 = fn(arg1, arg2)
        res2 = compiled(arg1, arg2)
        torch.testing.assert_close(res1, res2)
else:
    print("CUDA is not available. Skipping test as it requires CUDA inputs.")