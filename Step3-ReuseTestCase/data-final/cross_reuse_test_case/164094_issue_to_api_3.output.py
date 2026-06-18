import torch
import torch.cuda

class BackwardStreamExp(torch.autograd.Function):
    """
    This custom autograd function attempts to run the backward pass on a specific CUDA stream.
    It leverages the exponential function (similar to tf.experimental.numpy.exp) to perform
    actual computation, making the stream switching behavior observable.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        ctx.stream = stream
        # Save input for backward to compute gradient of exp
        ctx.save_for_backward(input_tensor)
        # Perform exp operation (leveraging the similar API pattern)
        return torch.exp(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        stream = ctx.stream
        # Attempt to switch streams for the backward pass
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(stream)
        
        # Gradient of exp(x) is exp(x) * grad_output
        input_tensor, = ctx.saved_tensors
        return torch.exp(input_tensor) * grad_output, None

def test_backward_stream_change():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Create a new stream
    s = torch.cuda.Stream()
    
    # Create a tensor requiring gradient
    x = torch.randn(10, device='cuda', requires_grad=True)
    
    # Run forward pass
    # Note: The bug report indicates that the backward op might not run on 's'
    y = BackwardStreamExp.apply(x, s)
    
    # Run backward pass
    y.sum().backward()
    
    # Basic sanity check to ensure the graph executed
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Note: Verifying the actual stream used by the backward kernel 
    # typically requires external profiling tools like Nsight Systems.
    # This test case reproduces the logic described in the issue.

if __name__ == "__main__":
    test_backward_stream_change()