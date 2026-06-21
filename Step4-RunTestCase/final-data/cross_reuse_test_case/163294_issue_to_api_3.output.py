import torch

# Leveraging the similar API pattern (trace) for inspection during the test.
# This mirrors the usage of tf.autograph.trace/tf.compat.v1.linalg.trace
# provided in the similar API information.
def trace(*args):
    """Traces argument information at compilation time."""
    print(*args)

class SetGradCase(torch.nn.Module):
    def forward(self, x):
        with torch.no_grad():
            y = x * 4
        return y

def test_no_grad_retrace_submodule_integrity():
    """
    Test that retracing an ExportedProgram containing a torch.no_grad context
    does not result in an empty submodule (Issue 163294).
    """
    # Fix: Check if torch.export exists (available in PyTorch 2.1+)
    if not hasattr(torch, 'export'):
        print("Skipping test: torch.export is not available in this PyTorch version (requires PyTorch >= 2.1).")
        return

    # First export
    ep = torch.export.export(
        SetGradCase(),
        (torch.randn(6),),
        strict=False,
    )
    trace("First Export:", ep)

    # Second export (retracing)
    # Bug: ep2's submod_1 becomes empty here.
    ep2 = torch.export.export(ep.module(), (torch.randn(6),))
    trace("Second Export:", ep2)

    # Verify the fix: The submodule should contain the operations inside torch.no_grad
    submod = ep2.graph_module.submod_1
    
    # Collect actual operations (excluding placeholders and output nodes)
    actual_ops = [n for n in submod.graph.nodes if n.op in ("call_function", "call_method")]
    
    # Assert that the submodule is not empty
    assert len(actual_ops) > 0, (
        f"Bug reproduced: submod_1 is empty after retrace. "
        f"Expected operations, but found {len(actual_ops)}."
    )
    
    # Verify the specific operation (mul) exists
    has_mul = any("mul" in str(n.target) for n in actual_ops)
    assert has_mul, "Expected 'mul' operation in submod_1, but it was not found."

if __name__ == "__main__":
    test_no_grad_retrace_submodule_integrity()