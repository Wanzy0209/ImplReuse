import torch
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    A: torch.Tensor
    k: int

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def fn(tup: MyNamedTuple) -> MyNamedTuple:
    # Add dynamic attribute (Bug context)
    extra_info = torch.tensor(4.0)
    tup.extra_info = extra_info
    
    # Call the similar API: torch.lobpcg
    # We use the tensor 'A' from the tuple as input
    # Note: lobpcg requires a symmetric positive definite matrix
    eigenvalues, eigenvectors = torch.lobpcg(tup.A, k=tup.k)
    
    return tup

# Setup valid inputs for torch.lobpcg
# Create a symmetric positive definite matrix
# Fix: Increase n to 10 to satisfy the constraint n >= 3 * k (10 >= 3 * 2)
n = 10
k = 2
A_tensor = torch.randn(n, n)
A_tensor = A_tensor @ A_tensor.T + torch.eye(n) * 0.1

extended_tup = MyNamedTupleSubclass(A=A_tensor, k=k)

print("\nTesting NamedTuple with __setattr__ and torch.lobpcg (Eager):")
eager_result = fn(extended_tup)
print(f"Eager extra_info: {eager_result.extra_info}")
assert eager_result.extra_info is not None, "Eager mode failed to persist attribute"

print("\nTesting NamedTuple with __setattr__ and torch.lobpcg (Compiled):")
try:
    compiled_fn = torch.compile(fn, backend="eager")
    compiled_result = compiled_fn(extended_tup)
    print(f"Compiled extra_info: {compiled_result.extra_info}")
    
    # This assertion will fail if the bug (Issue 161610) is present
    assert compiled_result.extra_info is not None, "Compiled mode failed to persist attribute"
    print("Test Passed: Attribute persisted in compiled mode.")
except AttributeError as e:
    print(f"Test Failed: {e}")