import torch

# Setup data for the scoping issue
keys = range(10)
allowed = [0, 1, 2, 3]

def fn(x):
    x = x + 1
    
    # Leverage the similar API: torch.backends.nnpack.set_flags
    # This replaces torch._dynamo.graph_break from the original issue reproduction.
    # We use it here to test if the codegen error affects functions calling this API.
    torch.backends.nnpack.set_flags(True)

    # This list comprehension creates the 'local vs cell' name conflict
    # that triggered the KeyError in torch._dynamo's fix_vars.
    key = [key for key in keys if key in allowed]

    def inner():
        nonlocal key

    # Restore state
    torch.backends.nnpack.set_flags(False)

    return x + key[0]

# Check if torch.compile is available (requires PyTorch 2.0+)
if hasattr(torch, 'compile'):
    # The bug was that torch.compile would crash with InternalTorchDynamoError
    # when processing the bytecode of 'fn'. This test verifies that the fix
    # works even when the similar API (set_flags) is involved.
    compiled_fn = torch.compile(fn, backend="eager")
    result = compiled_fn(torch.ones(3))
else:
    # Fallback for older PyTorch versions: run the function directly.
    # Note: This does not test the dynamo compilation bug, but ensures
    # the script runs without crashing on the AttributeError.
    print("torch.compile is not available. Running function without compilation.")
    result = fn(torch.ones(3))

# Assert the result is correct (x=1 -> x+1=2, key[0]=0 -> 2+0=2)
assert torch.allclose(result, torch.tensor([2., 2., 2.]))