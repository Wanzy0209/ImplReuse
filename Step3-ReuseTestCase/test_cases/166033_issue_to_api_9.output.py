import torch
import torch.nn.functional as F

def test_conv_transpose2d_dynamo_graph_break():
    # Setup inputs for conv_transpose2d
    # Input: (Batch=1, Channels=3, Height=5, Width=5)
    inp = torch.randn(1, 3, 5, 5)
    # Weight: (in_channels=3, out_channels=1, kH=3, kW=3)
    weight = torch.randn(3, 1, 3, 3)

    flag = True
    dummy = lambda: None

    def fn(x, w):
        # Use the similar API (conv_transpose2d) instead of simple addition
        x = F.conv_transpose2d(x, w)
        
        # The critical graph break from the original issue
        torch._dynamo.graph_break()
        
        # Use the similar API again
        x = F.conv_transpose2d(x, w)
        
        # The problematic conditional logic involving torch.no_grad
        if flag:
            dummy.attr0 = x
        else:
            with torch.no_grad():
                dummy.attr1 = x
        return x

    # Compile the function
    opt_fn = torch.compile(fn, backend="eager")

    # First run with flag = True
    assert torch.allclose(fn(inp, weight), opt_fn(inp, weight))

    # Second run with flag = False
    # This change in control flow triggers the bytecode transformation logic
    # that caused the KeyError in the original issue.
    flag = False
    assert torch.allclose(fn(inp, weight), opt_fn(inp, weight))

if __name__ == "__main__":
    test_conv_transpose2d_dynamo_graph_break()