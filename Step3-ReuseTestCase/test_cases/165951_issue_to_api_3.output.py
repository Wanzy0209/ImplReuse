import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import tracing, TracingContext
from contextlib import ExitStack

# Define a module that leverages the similar API (torch.special.expit)
# inside its forward pass, while also accepting its own kwargs.
class ModuleWithExpitAndKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x, scale=1.0):
        # Use torch.special.expit (similar API) which has signature: expit(input, *, out=None)
        # This tests the interaction between the module's kwargs and the kwargs of internal ops.
        linear_out = self.linear(x)
        expit_out = torch.special.expit(linear_out)
        return expit_out * scale

# Helper functions from the original bug report
def graph_capture_and_aot_export_joint_with_descriptors(model, inputs, kwargs=None):
    if kwargs is None:
        kwargs = {}
    with torch._dynamo.config.patch(install_free_tensors=True):
        # TODO: switch to use the official graph_capture API once it is ready
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

def test_aot_export_with_expit_and_kwargs():
    """
    Test case to verify that aot_export_joint_with_descriptors works with kwargs
    when the module uses torch.special.expit.
    """
    model = ModuleWithExpitAndKwargs()
    
    inputs = (torch.randn(4, 3),)
    # The bug specifically involves passing kwargs to the export mechanism
    kwargs = {"scale": torch.randn(1)}

    # Execute the export pipeline
    # This should not raise an error if the bug is fixed
    gm = graph_capture_and_aot_export_joint_with_descriptors(model, inputs, kwargs=kwargs)

    # Basic assertion to ensure the graph module was created
    assert gm is not None
    assert isinstance(gm, torch.fx.GraphModule)
    
    print("Test passed: aot_export_joint_with_descriptors handles kwargs with torch.special.expit.")

if __name__ == "__main__":
    test_aot_export_with_expit_and_kwargs()