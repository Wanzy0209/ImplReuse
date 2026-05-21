import torch

def test_masked_select_with_graph_break():
    """
    Test case derived from Issue 166033, adapted to use torch.masked_select.
    The original issue involved a KeyError in bytecode_transformation when
    using torch.no_grad inside a conditional block after a graph_break.
    This test checks if torch.masked_select handles a similar control flow
    pattern correctly under torch.compile.
    """
    flag = True
    dummy = lambda: None

    def fn(x):
        x = x + 1
        torch._dynamo.graph_break()
        x = x + 2
        
        if flag:
            # Use masked_select in the first branch
            mask = x > 0
            dummy.attr0 = torch.masked_select(x, mask)
        else:
            # Use masked_select in the second branch
            # Retaining the context manager structure to test interaction
            with torch.no_grad():
                mask = x > 0
                dummy.attr1 = torch.masked_select(x, mask)
                
        return x + 4

    inp = torch.ones(3)
    opt_fn = torch.compile(fn, backend="eager")
    
    # First run with flag = True
    assert torch.allclose(fn(inp), opt_fn(inp))
    
    # Second run with flag = False to trigger the alternative path
    # This is where the original KeyError occurred
    flag = False
    assert torch.allclose(fn(inp), opt_fn(inp))

if __name__ == "__main__":
    test_masked_select_with_graph_break()