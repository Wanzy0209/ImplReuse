import torch
import torch.nn as nn
import unittest

class ReflectionPadStreamSwitch(torch.autograd.Function):
    """
    A custom autograd Function that applies ReflectionPad1d in the forward pass
    and attempts to switch CUDA streams in the backward pass.
    This reproduces the logic from Issue 164094 using the torch.nn.ReflectionPad1d API.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        # Leverage the similar API: torch.nn.ReflectionPad1d
        # We instantiate the module with a padding of 2
        pad_layer = nn.ReflectionPad1d(2)
        
        ctx.stream = stream
        # Apply the padding operation
        return pad_layer(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        stream = ctx.stream
        # Original bug reproduction logic: attempt to switch streams for backward
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(stream)
        
        # Return gradients for input_tensor and None for the stream argument
        return grad_output, None

class TestReflectionPadBackwardStream(unittest.TestCase):
    def test_reflection_pad_backward_stream_change(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a dummy input tensor (Batch, Channel, Width)
        # ReflectionPad1d pads the last dimension
        input_tensor = torch.randn(2, 3, 10, device='cuda', requires_grad=True)
        
        # Create a secondary CUDA stream
        s = torch.cuda.Stream()
        
        # Apply the custom function
        # Forward pass runs on default stream, but we pass 's' for backward
        output = ReflectionPadStreamSwitch.apply(input_tensor, s)
        
        # Verify forward pass output shape (Width + padding_left + padding_right)
        # 10 + 2 + 2 = 14
        self.assertEqual(output.shape, (2, 3, 14))
        
        # Perform backward pass
        # This triggers the backward method where we attempt to switch to stream 's'
        output.sum().backward()
        
        # Verify gradients were computed
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)

if __name__ == '__main__':
    unittest.main()