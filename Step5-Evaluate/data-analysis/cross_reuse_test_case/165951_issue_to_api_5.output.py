import torch
import torch.nn as nn
from contextlib import ExitStack

# Imports based on the bug report
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import tracing, TracingContext

def test_aot_export_joint_with_descriptors_kwargs():
    """
    Test case to verify that aot_export_joint_with_descriptors works correctly
    when the model forward pass uses kwargs and kwargs are provided during export.
    This addresses the issue where the combination of _dynamo_graph_capture_for_export
    and aot_export_joint_with_descriptors failed with kwargs.
    """
    class ModuleWithKwargs(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)

        def forward(self, x, scale=1.0):
            return self.linear(x) * scale

    model = ModuleWithKwargs()
    inputs = (torch.randn(4, 3),)
    kwargs = {"scale": torch.randn(1)}

    # Step 1: Capture the graph using dynamo
    with torch._dynamo.config.patch(install_free_tensors=True):
        gm = _dynamo_graph_capture_for_export(model)(*inputs, **kwargs)
        fake_mode = gm.meta.get("fake_mode", None)

    # Step 2: Export using aot_export_joint_with_descriptors
    # This step was failing in the original bug report
    with tracing(TracingContext(fake_mode)):
        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack,
                gm,
                inputs,
                kwargs=kwargs,
            )
            result_gm = joint_with_descriptors.graph_module

    # Assertions to verify the fix
    assert result_gm is not None, "Exported graph module should not be None"
    
    # Verify the graph can be executed (optional but good for robustness)
    # Note: Depending on the exact state of the export, inputs might need to be 
    # adapted, but usually, we just check that the export process completes without error.
    # The bug report implies a crash or error during the export call itself.
    
    print("Test passed: aot_export_joint_with_descriptors handles kwargs correctly.")

if __name__ == "__main__":
    test_aot_export_joint_with_descriptors_kwargs()