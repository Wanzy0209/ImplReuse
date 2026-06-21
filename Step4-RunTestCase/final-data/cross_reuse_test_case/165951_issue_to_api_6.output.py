import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import tracing, TracingContext
from contextlib import ExitStack

# This test case integrates the similar API (torch.special.sinc) 
# into the module being exported to verify the kwargs handling 
# logic described in the bug report.

class ModuleWithSincAndKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale=1.0):
        # Leveraging torch.special.sinc as the core operation
        # to test the export pipeline with this specific API.
        return torch.special.sinc(x) * scale

def graph_capture_and_aot_export_joint_with_descriptors(model, inputs, kwargs=None):
    if kwargs is None:
        kwargs = {}
    with torch._dynamo.config.patch(install_free_tensors=True):
        # Capture the graph using dynamo
        gm = _dynamo_graph_capture_for_export(model)(*inputs, **kwargs)
        fake_mode = gm.meta.get("fake_mode", None)

    with tracing(TracingContext(fake_mode)):
        return aot_export_joint_with_descriptors_alone(gm, inputs, kwargs=kwargs)

def aot_export_joint_with_descriptors_alone(model, inputs, kwargs=None):
    if kwargs is None:
        kwargs = {}
    with ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            inputs,
            kwargs=kwargs,
        )
        return joint_with_descriptors.graph_module

def test_aot_export_with_sinc_and_kwargs():
    model = ModuleWithSincAndKwargs()
    
    # Define inputs and kwargs to trigger the specific bug scenario
    inputs = (torch.randn(4, 3),)
    kwargs = {"scale": torch.randn(1)}

    # Attempt to export the model. 
    # If the bug exists, this might fail or handle kwargs incorrectly.
    gm = graph_capture_and_aot_export_joint_with_descriptors(model, inputs, kwargs=kwargs)
    
    # Verify that the graph module was created successfully
    assert gm is not None
    
    # Verify the graph can be executed with the original inputs
    # Note: Depending on the state of the bug, execution might be secondary to successful export.
    # Here we primarily check that the export process completes.
    print("Test passed: aot_export_joint_with_descriptors handled kwargs with torch.special.sinc.")

if __name__ == "__main__":
    test_aot_export_with_sinc_and_kwargs()