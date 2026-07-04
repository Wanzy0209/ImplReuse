import torch
import torch.cuda

class BackwardStreamExp(torch.autograd.Function):
    """
    Custom autograd function that attempts to switch CUDA streams during the backward pass.
    This test case leverages the 'exp' operation (similar to tf.keras.ops.exp) 
    within the forward pass to provide a concrete computational context for the stream issue.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        ctx.stream = stream
        # Leveraging the similar API's functionality (exponential) using torch.exp
        return torch.exp(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        stream = ctx.stream
        # The reported bug: attempting to change the stream here may not affect 
        # the actual execution of backward operations.
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(stream)
        
        # For the purpose of this test, we return the gradient directly.
        # In a real scenario for exp(x), we would return exp(x) * grad_output.
        return grad_output, None

def test_backward_stream_change_with_exp():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Setup input tensor and a separate CUDA stream
    x = torch.randn(10, device='cuda', requires_grad=True)
    s = torch.cuda.Stream()

    # Run forward pass
    # The forward op (exp) runs on the current (default) stream
    y = BackwardStreamExp.apply(x, s)

    # Run backward pass
    # The backward op attempts to run on stream 's'
    y.sum().backward()

    # Assertions
    assert x.grad is not None, "Gradient computation failed"
    print("Test passed: Backward stream logic executed.")

if __name__ == "__main__":
    test_backward_stream_change_with_exp()