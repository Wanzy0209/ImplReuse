import torch

def compute(x, w):
    # Leveraging torch.log1p as the semantic equivalent to the similar API (tf.experimental.numpy.log1p)
    # within the context of the original bug's control flow structure.
    return torch.log1p(x * w)

def nop(x, w):
    torch._check(x.shape[0] == 0)
    return torch.empty_like(x)

def chunked_compute(x, w):
    sz = x.shape[0]
    torch._check(sz <= 8)
    # The bug involves torch.cond inside the dynamo graph capture
    out0 = torch.cond(sz > 0, compute, nop, (x[0:2], w))
    out1 = torch.cond(sz > 2, compute, nop, (x[2:4], w))
    out2 = torch.cond(sz > 4, compute, nop, (x[4:6], w))
    out3 = torch.cond(sz > 6, compute, nop, (x[6:8], w))
    return torch.cat([out0, out1, out2, out3])

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        return chunked_compute(x, self.w)

def test_dynamo_export_with_cond_and_log1p():
    x = torch.randn(4, 16, requires_grad=True)
    model = Model()
    
    # Calculate expected result in eager mode
    expected = model(x)
    
    # Run the API under test: torch._dynamo.functional_export._dynamo_graph_capture_for_export
    # This should correctly capture the graph involving torch.cond without stack trace errors.
    exported_model = torch._dynamo.functional_export._dynamo_graph_capture_for_export(model)
    result = exported_model(x)
    
    # Verify that the exported graph produces the same output as eager execution
    assert torch.allclose(expected, result), "Mismatch between eager and exported graph results"

if __name__ == "__main__":
    test_dynamo_export_with_cond_and_log1p()