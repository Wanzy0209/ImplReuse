import torch
import torch.library
import sys

# Fix: Handle import for different PyTorch versions
# torch.export was introduced in PyTorch 2.1. In 2.0 it was torch._export.
try:
    from torch.export import export
except ModuleNotFoundError:
    try:
        from torch._export import export
    except ImportError:
        print("Skipping test: torch.export module not found.")
        sys.exit(0)

def test_register_autograd_with_export():
    # Define a custom operator
    torch.library.define("test_ns::custom_mul", "(Tensor x, Tensor y) -> Tensor")

    # Register the implementation for CPU
    def custom_mul_impl(x, y):
        return x * y

    torch.library.impl("test_ns::custom_mul", "cpu", custom_mul_impl)

    # Register the autograd function using the similar API: torch.library.register_autograd
    def custom_mul_backward(ctx, grad_x, grad_y):
        x, y = ctx.saved_tensors
        # Gradient of x * y w.r.t x is y, w.r.t y is x
        return grad_x * y, grad_y * x

    def custom_mul_setup(ctx, x, y):
        ctx.save_for_backward(x, y)

    torch.library.register_autograd(
        "test_ns::custom_mul",
        custom_mul_backward,
        setup_context=custom_mul_setup
    )

    # Define a simple model using the custom op
    class CustomModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.ops.test_ns.custom_mul(x, y)

    # Prepare inputs
    model = CustomModel()
    example_inputs = (torch.randn(2, 2), torch.randn(2, 2))

    # Adapt the original call site: Attempt to export the model with the custom op
    # This verifies that torch.library.register_autograd allows the model to be exported
    # without raising "Current active mode not registered".
    try:
        ep = export(model, example_inputs)
        
        # Verify the exported program runs correctly
        output = ep.module()(*example_inputs)
        expected = example_inputs[0] * example_inputs[1]
        
        assert torch.allclose(output, expected), "Exported model output mismatch"
        print("Test passed: torch.library.register_autograd compatible with torch.export.export")
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")
        raise
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        raise

if __name__ == "__main__":
    test_register_autograd_with_export()