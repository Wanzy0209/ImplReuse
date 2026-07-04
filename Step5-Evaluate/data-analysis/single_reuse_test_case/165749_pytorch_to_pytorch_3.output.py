import torch
from torch import nn
from torch.optim import SGD

# Check for PyTorch 2.0+ features required for this test
HAS_TORCH_2 = False
try:
    from torch.library import Library, impl_abstract, impl
    HAS_TORCH_2 = True
except ImportError:
    print("Skipping test: PyTorch 2.0+ is required for torch.compile and torch.library.impl_abstract.")

if HAS_TORCH_2:
    # Define a custom library to mimic the behavior of the failing operator
    lib = Library("test_weight_norm", "DEF")

    # Define a custom operator that mimics the weight normalization forward pass
    # This operator returns the normalized weight and the original weight (v) for backward.
    lib.define("custom_weight_norm_fwd(Tensor v, Tensor g, int dim) -> (Tensor, Tensor)")

    # Use the similar API: torch.library.impl_abstract
    # This defines the behavior for FakeTensors (used by torch.compile) to infer output shapes and properties.
    @impl_abstract("test_weight_norm::custom_weight_norm_fwd")
    def custom_weight_norm_fwd_abstract(v, g, dim):
        # Infer the norm of v
        norm = v.norm(dim, keepdim=True)
        # Infer the normalized weight w
        w = g * (v / norm)
        # The bug report mentions "saved_v must be contiguous".
        # In the abstract implementation, we ensure the returned 'v' (saved for backward)
        # preserves the contiguity property of the input 'v'.
        # If the input v is contiguous, the output v should be marked as contiguous.
        return w, v

    # Define the concrete implementation for CPU and CUDA
    @impl("test_weight_norm::custom_weight_norm_fwd", "CPU")
    @impl("test_weight_norm::custom_weight_norm_fwd", "CUDA")
    def custom_weight_norm_fwd_impl(v, g, dim):
        # Enforce contiguity check similar to the C++ kernel in the bug report
        if not v.is_contiguous():
            raise RuntimeError("saved_v must be contiguous")
        
        norm = v.norm(dim, keepdim=True)
        w = g * (v / norm)
        return w, v

    # Define a custom backward operator to complete the autograd graph
    lib.define("custom_weight_norm_bwd(Tensor grad_w, Tensor v, Tensor g, Tensor w, int dim) -> (Tensor, Tensor)")

    @impl_abstract("test_weight_norm::custom_weight_norm_bwd")
    def custom_weight_norm_bwd_abstract(grad_w, v, g, w, dim):
        # Return shapes for gradients of v and g
        return v, g

    @impl("test_weight_norm::custom_weight_norm_bwd", "CPU")
    @impl("test_weight_norm::custom_weight_norm_bwd", "CUDA")
    def custom_weight_norm_bwd_impl(grad_w, v, g, w, dim):
        # Simplified backward logic for demonstration
        norm = v.norm(dim, keepdim=True)
        grad_v = grad_w * g / norm
        grad_g = (grad_w * v / norm).sum(dim=dim, keepdim=True)
        return grad_v, grad_g

    # Setup autograd.Function to use the custom ops
    class CustomWeightNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v, g, dim):
            w, v_saved = torch.ops.test_weight_norm.custom_weight_norm_fwd(v, g, dim)
            ctx.save_for_backward(v_saved, g, w)
            ctx.dim = dim
            return w

        @staticmethod
        def backward(ctx, grad_w):
            v, g, w = ctx.saved_tensors
            dim = ctx.dim
            grad_v, grad_g = torch.ops.test_weight_norm.custom_weight_norm_bwd(grad_w, v, g, w, dim)
            return grad_v, grad_g, None

    # Adapt the original reproducer to use the custom operator
    if __name__ == "__main__":
        d = 65
        x = torch.randn((1, 2, 32, 32)).cuda()
        
        # Create a model that applies the custom weight norm to a Conv2d weight
        class CustomWNModel(nn.Module):
            def __init__(self, in_c, out_c, k):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, k)
                # Initialize g parameter
                self.g = nn.Parameter(torch.ones(out_c, 1, 1, 1).cuda())
                
            def forward(self, x):
                v = self.conv.weight
                # Apply custom weight norm
                w = CustomWeightNormFunction.apply(v, self.g, 0)
                # Use the modified weight for convolution
                # Note: This is a simplified structural replacement for demonstration.
                # We manually perform the convolution with the modified weight.
                return torch.conv2d(x, w, self.conv.bias, self.conv.stride, 
                                    self.conv.padding, self.conv.dilation, self.conv.groups)

        # Original call site: model = torch.compile(weight_norm(nn.Conv2d(2, d, 2)).train().cuda())
        # Adapted call site: We compile the model using our custom operator which has the registered abstract impl.
        model = torch.compile(CustomWNModel(2, d, 2).train().cuda())
        
        opt = SGD(model.parameters(), lr=0.01)
        
        try:
            for _ in range(2):
                out = model(x)
                loss = out.mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
            print("Test passed: Backward pass succeeded with torch.compile and custom abstract impl.")
        except RuntimeError as e:
            print(f"Test failed: {e}")