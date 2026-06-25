import torch

def test_export_kwargs():
    class TestModule(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu(x, inplace=True)
    
    mod = TestModule()
    exported = torch.export.export(mod, (torch.randn(2, 2),))
    print(exported.graph)
    # Observe that 'inplace=True' becomes positional argument

test_export_kwargs()