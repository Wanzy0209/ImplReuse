import torch

def test_isinf_dynamo_graph_break():
    """
    Test case for torch.isinf based on the bytecode transformation bug pattern.
    Original issue (166033) involved a KeyError with torch.no_grad inside a conditional
    after a graph break. This test adapts that logic to use torch.isinf.
    """
    flag = True
    dummy = lambda: None

    def fn(x):
        x = x + 1
        torch._dynamo.graph_break()
        x = x + 2
        
        if flag:
            # Use torch.isinf in the first branch
            dummy.attr0 = torch.isinf(x)
        else:
            # Use torch.isinf inside the context manager in the second branch
            # This preserves the structure that caused the original bug
            with torch.no_grad():
                dummy.attr1 = torch.isinf(x)
        
        return x + 4

    inp = torch.ones(3)
    opt_fn = torch.compile(fn, backend="eager")
    
    # First run with flag = True
    expected_out = fn(inp)
    actual_out = opt_fn(inp)
    assert torch.allclose(expected_out, actual_out)
    # Verify the side effect of isinf (should be False for finite values)
    assert torch.allclose(dummy.attr0, torch.zeros(3, dtype=torch.bool))

    # Second run with flag = False (triggers recompilation/alternative path)
    flag = False
    expected_out = fn(inp)
    actual_out = opt_fn(inp)
    assert torch.allclose(expected_out, actual_out)
    # Verify the side effect of isinf inside no_grad
    assert torch.allclose(dummy.attr1, torch.zeros(3, dtype=torch.bool))

if __name__ == "__main__":
    test_isinf_dynamo_graph_break()