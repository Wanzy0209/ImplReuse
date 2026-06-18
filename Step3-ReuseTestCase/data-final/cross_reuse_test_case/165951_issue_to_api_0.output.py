import torch
import torch.nn as nn
from contextlib import ExitStack
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import tracing, TracingContext

def test_aot_export_joint_with_descriptors_kwargs():
    """
    Test that aot_export_joint_with_descriptors works correctly with kwargs.
    Regression test for Issue #165951.
    
    This test verifies the pipeline where a model with keyword arguments
    is captured via _dynamo_graph_capture_for_export and then passed to
    aot_export_joint_with_descriptors. The bug report indicated that
    passing kwargs through this specific chain would fail.
    """
    class ModuleWithKwargs(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x, scale=1.0):
            return self.linear(x) * scale

    model = ModuleWithKwargs()
    
    # Setup inputs and keyword arguments
    inputs = (torch.randn(4, 3),)
    kwargs = {"scale": torch.randn(1)}

    # Step 1: Capture the graph using dynamo
    with torch._dynamo.config.patch(install_free_tensors=True):
        gm = _dynamo_graph_capture_for_export(model)(*inputs, **kwargs)
        fake_mode = gm.meta.get("fake_mode", None)

    # Step 2: Export using aot_export_joint_with_descriptors with kwargs
    # This step was failing before the fix.
    with tracing(TracingContext(fake_mode)):
        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack,
                gm,
                inputs,
                kwargs=kwargs,
            )
            exported_gm = joint_with_descriptors.graph_module

    # Verify the export produced a valid graph module
    assert exported_gm is not None, "Exported graph module should not be None"
    
    # Verify that the graph module has the expected structure (basic check)
    # The graph should contain operations corresponding to the forward pass
    graph_nodes = [node.name for node in exported_gm.graph.nodes]
    assert "linear" in str(graph_nodes) or "addmm" in str(graph_nodes), \
        "Graph should contain linear operation nodes"

    print("Test passed: aot_export_joint_with_descriptors handles kwargs correctly.")

if __name__ == "__main__":
    test_aot_export_joint_with_descriptors_kwargs()