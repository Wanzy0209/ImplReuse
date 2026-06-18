import torch
from collections import namedtuple

def test_namedtuple_export():
    """
    Test case to verify torch.export.export behavior with NamedTuple inputs.
    Based on Issue ID: 160547
    """
    # Define a simple NamedTuple
    Point = namedtuple('Point', ['x', 'y'])
    
    # Define a simple Module
    class SimpleModule(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    
    # Create input using the NamedTuple
    inp = Point(torch.ones(3), torch.ones(3))
    
    # Verify the module runs normally with the input
    model = SimpleModule()
    output = model(*inp)
    assert torch.allclose(output, torch.ones(3) * 2), "Module forward pass failed"

    # Test Case 1: Non-strict export (The reported bug scenario)
    # This should ideally succeed or handle the input gracefully.
    try:
        ep_non_strict = torch.export.export(model, inp, strict=False)
        assert isinstance(ep_non_strict, torch.export.ExportedProgram)
        print("Test Case 1 Passed: Non-strict export with NamedTuple input succeeded.")
    except Exception as e:
        print(f"Test Case 1 Failed: Non-strict export with NamedTuple input raised an exception: {e}")
        raise

    # Test Case 2: Strict export (The working scenario in the bug report)
    try:
        ep_strict = torch.export.export(model, inp, strict=True)
        assert isinstance(ep_strict, torch.export.ExportedProgram)
        print("Test Case 2 Passed: Strict export with NamedTuple input succeeded.")
    except Exception as e:
        print(f"Test Case 2 Failed: Strict export with NamedTuple input raised an exception: {e}")
        raise

    # Test Case 3: Workaround using kwargs (from the bug report)
    try:
        inp_kwargs = {field: getattr(inp, field) for field in inp._fields}
        ep_kwargs = torch.export.export(model, (), inp_kwargs)
        assert isinstance(ep_kwargs, torch.export.ExportedProgram)
        print("Test Case 3 Passed: Export with NamedTuple converted to kwargs succeeded.")
    except Exception as e:
        print(f"Test Case 3 Failed: Export with kwargs raised an exception: {e}")
        raise

if __name__ == "__main__":
    test_namedtuple_export()