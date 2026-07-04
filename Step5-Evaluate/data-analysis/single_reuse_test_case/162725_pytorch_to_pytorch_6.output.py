import torch

def fn(input, mat, vec, beta=1, alpha=1):
    result = torch.addmv(input, mat, vec, beta=beta, alpha=alpha)
    return result

# The original test failed because torch.testing._internal.common_utils imports 'expecttest',
# which is not available in the environment. We fix this by manually generating sample inputs
# instead of relying on the internal op_db.

# Helper class to mimic the structure of op_db samples
class SampleInput:
    def __init__(self, input, args):
        self.input = input
        self.args = args

# Configuration matching the original test
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
requires_grad = False

# Manually create sample inputs for torch.addmv
# Signature: (input, mat, vec, beta=1, alpha=1)
# Constraints: mat is (m, n), vec is (n), input is (m)
inputs = [
    SampleInput(
        input=torch.randn(3, device=device, dtype=dtype, requires_grad=requires_grad),
        args=(
            torch.randn(3, 4, device=device, dtype=dtype, requires_grad=requires_grad), # mat
            torch.randn(4, device=device, dtype=dtype, requires_grad=requires_grad),    # vec
            1,  # beta
            1   # alpha
        )
    ),
    SampleInput(
        input=torch.randn(5, device=device, dtype=dtype, requires_grad=requires_grad),
        args=(
            torch.randn(5, 2, device=device, dtype=dtype, requires_grad=requires_grad), # mat
            torch.randn(2, device=device, dtype=dtype, requires_grad=requires_grad),    # vec
            0.5, # beta
            2.0  # alpha
        )
    )
]

for sample in inputs:
    # Prepare arguments: sample.input is the first arg, sample.args are the rest
    args = (sample.input, *sample.args)
    
    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
    
    res1 = fn(*args)
    res2 = compiled(*args)
    
    torch.testing.assert_close(res1, res2)