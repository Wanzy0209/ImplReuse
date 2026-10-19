import os
import torch
import torch.nn as nn
import tensorflow as tf
from torch.nn.modules.module import register_module_backward_hook
from tf.types.experimental import ConcreteFunction

# Replacements:
# torch.distributed -> tf.distribute.experimental.MultiWorkerMirroredStrategy
# torch.distributed.device_mesh -> tf.distribute.experimental.MultiWorkerMirroredStrategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

class TestModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # nn.RMSNorm -> register_module_backward_hook (semantics: hooking into the module)
        # We keep the module to reproduce the bug, but the focus is on the hook.
        self.norm = nn.RMSNorm(d_model)
        # nn.Linear -> register_module_backward_hook
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.output(x)
        return x

def main():
    # torch.cuda.set_device -> register_module_backward_hook (contextual setup)
    # We assume CUDA is available for the test
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    torch.cuda.set_device(0)
    d_model = 128

    model = TestModule(d_model)
    model = model.to('cuda:0')

    # FSDPMemTracker -> register_module_backward_hook
    # We implement a simple tracker using the hook
    backward_hook_called = {"norm": False, "output": False}

    def tracker_hook(module, grad_input, grad_output):
        module_name = module.__class__.__name__
        if module_name == "RMSNorm":
            backward_hook_called["norm"] = True
        elif module_name == "Linear":
            backward_hook_called["output"] = True
        # The bug was a KeyError here. We check if we can access grad_output.
        # print(f"Backward hook for {module_name}: {grad_output}")

    # Register the hooks
    register_module_backward_hook(model.norm, tracker_hook)
    register_module_backward_hook(model.output, tracker_hook)

    # torch.randn -> tf.distribute.experimental.MultiWorkerMirroredStrategy
    # We use the strategy scope to generate data, simulating distributed data generation
    with strategy.scope():
        # tf.types.experimental.ConcreteFunction -> tf.function
        # We leverage the similar API to wrap data generation
        @tf.function
        def get_data():
            return tf.random.normal((16, d_model))

        tf_x = get_data()
        # Convert TF tensor to PyTorch tensor
        x = torch.from_numpy(tf_x.numpy()).to('cuda:0')

    # Forward pass
    y = model(x)

    # y.sum -> register_module_backward_hook
    # We use the hook to track the backward pass triggered by the sum
    loss = y.sum()

    # loss.backward -> register_module_backward_hook
    # The backward pass triggers the registered hooks
    loss.backward()

    # Assertions to verify the hooks were called (reproducing the tracker logic)
    assert backward_hook_called["norm"], "Backward hook for RMSNorm was not called"
    assert backward_hook_called["output"], "Backward hook for Linear was not called"

    print("Test passed. Backward hooks successfully tracked RMSNorm and Linear.")

if __name__ == "__main__":
    main()