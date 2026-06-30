import torch

# Test case for Issue 162313 adapted to use torch.no_grad
# This verifies that the resume execution logic handles context managers correctly
# when graph breaks occur within conditional branches that change state.

flag = True


@torch.compile(backend="eager")
def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 2
    if flag:
        # Using torch.no_grad() as a standard context manager
        # to test interaction with Dynamo's resume execution logic.
        with torch.no_grad():
            torch._dynamo.graph_break()
    else:
        with torch.no_grad():
            torch._dynamo.graph_break()
    return x + 4


# First call with flag = True
result1 = fn(torch.ones(3))
# Expected: (1 + 1) + 2 + 4 = 8
assert torch.all(result1 == 8), f"Expected 8, got {result1}"

# Toggle flag to trigger recompilation/resume logic
flag = False

# Second call with flag = False
result2 = fn(torch.ones(3))
assert torch.all(result2 == 8), f"Expected 8, got {result2}"

print("Test passed.")