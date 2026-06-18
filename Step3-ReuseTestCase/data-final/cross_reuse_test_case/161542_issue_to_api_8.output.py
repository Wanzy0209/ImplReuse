import torch

# Setup data for the list comprehension logic
keys = range(10)
allowed = [0, 1, 2, 3]

def fn(x):
    x = x + 1
    
    # Leverage the similar API: torch.backends.cusparselt.is_available
    # We integrate this check into the function to test if the dynamo compilation
    # handles backend availability checks correctly alongside the variable scoping issue.
    is_cusparse = torch.backends.cusparselt.is_available()

    # The graph break is essential to trigger the specific codegen path in the bug report
    torch._dynamo.graph_break()

    # The problematic pattern: 'key' is assigned via list comprehension (local scope)
    key = [key for key in keys if key in allowed]

    def inner():
        # 'nonlocal' forces 'key' to be treated as a cell variable in 'fn'
        nonlocal key

    # Return value combines the tensor operation, the list comp result, and the API result
    return x + key[0] + (1 if is_cusparse else 0)

# Compile and run the function
# The bug report indicates this raises torch._dynamo.exc.InternalTorchDynamoError
# This test verifies that the compilation succeeds and the logic is preserved.
try:
    compiled_fn = torch.compile(fn, backend="eager")
    result = compiled_fn(torch.ones(3))
    
    # Calculate expected value
    # x starts as 1.0, x + 1 = 2.0
    # key[0] is 0
    # is_cusparse is usually False, so + 0
    expected_val = 2.0 + (1 if torch.backends.cusparselt.is_available() else 0)
    
    assert torch.allclose(result, torch.full((3,), expected_val))
    print("Test Passed: Compilation succeeded and output is correct.")

except Exception as e:
    print(f"Test Failed: {e}")