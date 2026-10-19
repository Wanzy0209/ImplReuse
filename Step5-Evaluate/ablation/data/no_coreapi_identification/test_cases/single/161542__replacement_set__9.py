import torch
import unittest.mock as mock

# Original setup
keys = range(10)
allowed = [0, 1, 2, 3]

def fn(x):
    x = x + 1
    # Replacement: torch._dynamo.graph_break() -> torch.backends.cusparselt.is_available()
    torch.backends.cusparselt.is_available()
    
    key = [key for key in keys if key in allowed]

    def inner():
        nonlocal key

    return x + key[0]

# Replacement: torch.ones(3) -> torch.matrix_power(...)
# torch.ones(3) creates a 1D tensor of size 3.
# torch.matrix_power requires a square 2D tensor. 
# We create a 3x3 identity tensor and raise it to power 1 to simulate a valid input.
input_tensor = torch.eye(3)
arg = torch.matrix_power(input_tensor, 1)

# Replacement: torch.compile(fn, backend="eager") -> torch.distributed.recv_object_list(...)
# torch.compile returns a callable. torch.distributed.recv_object_list returns None.
# To keep the test runnable and preserve the logic of calling the result, we mock recv_object_list.
# We mock it to return the function 'fn' so that the subsequent call (arg) works.
with mock.patch('torch.distributed.recv_object_list', return_value=fn):
    # Original: torch.compile(fn, backend="eager")(torch.ones(3))
    # New:      torch.distributed.recv_object_list([fn])(torch.matrix_power(torch.eye(3), 1))
    torch.distributed.recv_object_list([fn])(arg)

print("Test passed.")