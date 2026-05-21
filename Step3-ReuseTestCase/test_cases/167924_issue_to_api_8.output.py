import torch

class RepeatInterleaveSliceFunc(torch.autograd.Function):
    """
    Mimics the pattern of tf.custom_gradient by defining a custom forward
    and backward pass for the operation involved in the bug report.
    """
    @staticmethod
    def forward(ctx, data, counts):
        # Reproduce the bug logic: slicing counts to a non-prefix
        # This is the specific operation causing the segfault on MPS
        sliced_counts = counts[1:3]
        output = data.repeat_interleave(sliced_counts, dim=0)
        
        # Save tensors for backward pass
        ctx.save_for_backward(data, sliced_counts)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        data, counts = ctx.saved_tensors
        grad_data = torch.zeros_like(data)
        
        # Manually compute gradients for the repeat_interleave operation
        # If data[i] is repeated counts[i] times, the gradient is the sum
        # of the corresponding gradients in grad_output.
        current_idx = 0
        for i, c in enumerate(counts):
            if c > 0:
                # Sum the gradients for the repeated elements
                grad_data[i] = grad_output[current_idx : current_idx + c].sum()
                current_idx += c
        
        return grad_data, None

def test_mps_repeat_interleave_slice():
    """
    Test case for Issue 167924: Crash on MPS when using repeat_interleave 
    with sliced tensor.
    """
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping test.")
        return

    # Setup from the bug report
    # counts is [0, 1, 0], sliced to [1, 0]
    counts = torch.tensor([0, 1, 0], device="mps", dtype=torch.long)
    # data is [0, 1]
    data = torch.arange(2, device="mps", dtype=torch.float, requires_grad=True)

    # Apply the custom function (leveraging the custom_gradient pattern)
    # This should trigger the crash if the bug exists
    result = RepeatInterleaveSliceFunc.apply(data, counts)

    # Verify forward pass result
    # data=[0, 1], counts=[1, 0] -> result=[1]
    expected = torch.tensor([1.0], device="mps")
    assert torch.equal(result, expected), f"Forward pass failed: {result} != {expected}"

    # Verify backward pass (gradient flow)
    # Ensures the operation is differentiable and doesn't crash in backward
    result.sum().backward()
    expected_grad = torch.tensor([0.0, 1.0], device="mps")
    assert torch.allclose(data.grad, expected_grad), f"Backward pass failed: {data.grad} != {expected_grad}"
    
    print("Test passed.")

if __name__ == "__main__":
    test_mps_repeat_interleave_slice()