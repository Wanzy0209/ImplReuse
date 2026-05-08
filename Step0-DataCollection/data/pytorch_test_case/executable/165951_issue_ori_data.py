from contextlib import ExitStack

import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._guards import tracing, TracingContext

class ModuleWithKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x, scale=1.0):
        return self.linear(x) * scale

model = ModuleWithKwargs()

inputs = (torch.randn(4, 3),)
kwargs = {"scale": torch.randn(1)}

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

gm = graph_capture_and_aot_export_joint_with_descriptors(model, inputs, kwargs=kwargs)