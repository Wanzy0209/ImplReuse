import torch
import sys

def test_retracing_set_grad_with_trace():
    """
    Test for Issue 163294: Retracing set_grad HOO creates an empty submod.
    This test leverages torch.trace (similar to tf.linalg.trace) inside the no_grad block
    to verify that the submodule retains its operations during re-exporting.
    """
    # Handle environment compatibility: torch.export is available in PyTorch 2.1+
    if not hasattr(torch, 'export'):
        print(f"SKIP: torch.export not available (requires PyTorch >= 2.1). Current version: {torch.__version__}")
        return

    class SetGradCase(torch.nn.Module):
        def forward(self, x):
            with torch.no_grad():
                # Using torch.trace to leverage the similar API pattern
                y = torch.trace(x)
            return y

    # First export
    ep = torch.export.export(
        SetGradCase(),
        (torch.randn(3, 3),), # 2D tensor required for trace
        strict=False,
    )

    # Second export (retracing) - This triggers the bug where the submod becomes empty
    ep2 = torch.export.export(ep.module(), (torch.randn(3, 3),))

    # Verify the structure of ep2
    # Find the wrap_with_set_grad_enabled node
    set_grad_node = None
    for node in ep2.graph.nodes:
        if node.target == torch.ops.higher_order.wrap_with_set_grad_enabled:
            set_grad_node = node
            break
    
    assert set_grad_node is not None, "wrap_with_set_grad_enabled node not found in re-exported graph"

    # Identify the submodule argument (usually the second arg: enabled, submod, inputs...)
    submod_node = set_grad_node.args[1]
    
    if submod_node.op == 'get_attr':
        submod_name = submod_node.name
        submod = ep2.graph_module.get_submodule(submod_name)
        
        # Check if submod is empty (the bug)
        # Filter out placeholder, output, and get_attr nodes to find actual computation
        ops = [n for n in submod.graph.nodes if n.op not in ('placeholder', 'output', 'get_attr')]
        
        assert len(ops) > 0, f"Submodule '{submod_name}' is empty in re-exported program (Issue 163294)."
        
        # Verify that the 'trace' operation is present inside the submodule
        has_trace = any('trace' in str(n.target) for n in ops)
        assert has_trace, f"Submodule '{submod_name}' is missing expected 'trace' operation."

    # Run the graph to ensure functional correctness
    inp = torch.randn(3, 3)
    res = ep2.module()(inp)
    expected = torch.trace(inp)
    assert torch.allclose(res, expected), "Output mismatch after re-exporting"

if __name__ == "__main__":
    test_retracing_set_grad_with_trace()