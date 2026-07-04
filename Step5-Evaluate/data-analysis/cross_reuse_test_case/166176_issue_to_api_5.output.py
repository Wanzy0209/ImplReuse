import torch

def fn(x):
    # First graph break to trigger resume logic
    torch._dynamo.graph_break()
    
    with torch.no_grad():
        # Leverage the similar API: torch.unbind
        # This introduces shape evaluation and list handling logic
        # similar to the implementation details provided.
        unbound_tensors = torch.unbind(x, dim=0)
        
        with torch.no_grad():
            # Second graph break inside nested context
            # This is the specific pattern that caused the KeyError in the issue.
            torch._dynamo.graph_break()
            
            # Perform an operation on the unbound tensors to ensure they are part of the graph
            # and not optimized away immediately.
            _ = [t + 1 for t in unbound_tensors]
            
    return x + 1

if __name__ == "__main__":
    inp = torch.ones(3)
    # Use "eager" backend as per the original bug report
    opt_m = torch.compile(fn, backend="eager")
    
    # Run the compiled model. 
    # If the bug is present, this will raise a KeyError in resume_execution.
    # If the bug is fixed, it will execute successfully.
    result = opt_m(inp)
    
    # Assertion to verify correctness
    expected = torch.ones(3) + 1
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
    print("Test passed.")