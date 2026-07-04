import torch
import unittest

class TestBackwardStreamChange(unittest.TestCase):
    def test_set_stream_in_backward(self):
        """
        Test case to reproduce the issue where torch.cuda.set_stream() 
        fails to change the stream during the backward pass.
        
        This test leverages the pattern from tf.compat.v1.local_variables_initializer
        by checking the execution context (CUDA availability) before proceeding.
        """
        # Leverage the similar API's pattern: check context/state before execution
        # tf.compat.v1.local_variables_initializer checks context.executing_eagerly()
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping stream test")

        class BackwardStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
                ctx.stream = stream
                return input_tensor

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                stream = ctx.stream
                # Ensure the new stream waits for the current one
                stream.wait_stream(torch.cuda.current_stream())
                
                # Attempt to switch stream
                torch.cuda.set_stream(stream)
                
                # Verify the stream has been switched in the current context
                # This assertion captures the user's expectation described in the bug report.
                current_stream = torch.cuda.current_stream()
                self.assertEqual(current_stream, stream, 
                                 "Failed to change backward stream: current stream does not match target stream")
                
                # Return gradient for input_tensor and None for the stream argument
                return grad_output, None

        # Setup inputs
        s = torch.cuda.Stream()
        x = torch.randn(10, 10, device='cuda', requires_grad=True)
        
        # Forward pass
        y = BackwardStream.apply(x, s)
        
        # Backward pass (triggers the backward method)
        y.sum().backward()

if __name__ == '__main__':
    unittest.main()