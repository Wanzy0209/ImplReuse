import torch
import torch.library
from torch.export import export

def test_custom_autograd_slice_export():
    """
    Test case based on Issue 163146: [export] Data dependent error on slices in autograd::applySlicing
    
    This test verifies the behavior of torch.library.register_autograd when used to define
    a custom slicing operation that is data-dependent, followed by a torch.export.export call.
    """

    # 1. Define a custom operator that mimics the dynamic slicing behavior
    torch.library.define("test_pkg::dynamic_slice", "(Tensor x, Tensor limit) -> Tensor")

    # 2. Implement the forward pass for the custom operator
    @torch.library.impl("test_pkg::dynamic_slice", "CPU")
    def dynamic_slice_cpu(x, limit):
        # Replicates the logic: item_embedding[:, :max_item_num, :]
        # Note: .item() introduces data dependency which is the source of the bug.
        return x[:, :limit.item(), :]

    # 3. Define the backward pass and setup context
    def dynamic_slice_backward(ctx, grad_output):
        x, limit = ctx.saved_tensors
        # Reconstruct the gradient for the input tensor
        grad_x = torch.zeros_like(x)
        grad_x[:, :limit.item(), :] = grad_output
        return grad_x, None

    def setup_context(ctx, x, limit):
        ctx.save_for_backward(x, limit)

    # 4. Register the autograd function using the similar API
    torch.library.register_autograd(
        "test_pkg::dynamic_slice",
        backward=dynamic_slice_backward,
        setup_context=setup_context
    )

    # 5. Define a model using the custom operator
    class SliceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(100, 64)

        def forward(self, indices, max_item_num):
            x = self.embedding(indices)
            # Use the custom op instead of standard slicing
            return torch.ops.test_pkg.dynamic_slice(x, max_item_num)

    # 6. Prepare inputs
    model = SliceModel()
    # Shape [10, 64] corresponding to s10, s64 in the bug report
    indices = torch.randint(0, 100, (10, 64)) 
    # Scalar tensor for the slice limit
    max_item_num = torch.tensor(50)

    # 7. Attempt to export the model
    # The original bug report indicates a "Data dependent error" during export.
    # We test if using a custom autograd op changes this behavior.
    try:
        ep = export(model, args=(indices, max_item_num))
        print("Export succeeded.")
        
        # Verify execution
        res = ep.module()(indices, max_item_num)
        assert res.shape == (10, 50, 64), f"Expected shape (10, 50, 64), got {res.shape}"
        print("Test Passed: Export and execution successful.")
        
    except Exception as e:
        print(f"Export failed with error: {e}")
        # Depending on the fix status, this might be expected or not.
        # For the purpose of this test generation, we capture the behavior.
        raise

if __name__ == "__main__":
    test_custom_autograd_slice_export()