import torch

def test_tril_dynamic():
    class TestModule(torch.nn.Module):
        def forward(self, x):
            return torch.tril(x)
    
    x = torch.randn(4, 4)
    dynamic_shapes = {'x': {0: torch.export.Dim('dim0'), 1: torch.export.Dim('dim1')}}
    exported = torch.export.export(TestModule(), (x,), dynamic_shapes=dynamic_shapes)
    return exported

test_tril_dynamic()