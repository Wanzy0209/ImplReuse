import torch
import tensorflow as tf

def test_issue_167924_replacement():
    """
    Test case based on Issue 167924: Crash on MPS when using repeat_interleave with sliced tensor.
    
    Replacements applied:
    - torch.tensor -> torch.empty_strided
    - repeat_interleave -> tf.custom_gradient
    """
    
    # Original: counts = torch.tensor([0, 1, 0], device="mps")
    # Replacement: torch.tensor -> torch.empty_strided
    # We create the tensor with empty_strided and then initialize it to preserve the original data logic.
    counts = torch.empty_strided((3,), (1,), dtype=torch.long, device="mps")
    with torch.no_grad():
        counts[0] = 0
        counts[1] = 1
        counts[2] = 0

    # Original: data = torch.arange(2, device="mps")
    data = torch.arange(2, device="mps")

    # Original: data.repeat_interleave(counts[1:3], dim=0)
    # Replacement: repeat_interleave -> tf.custom_gradient
    # We define a custom gradient function to mimic the repeat_interleave logic.
    
    @tf.custom_gradient
    def custom_repeat_op(x, repeats):
        # Forward pass: mimic repeat_interleave using tf.repeat
        # x corresponds to 'data', repeats corresponds to 'counts[1:3]'
        y = tf.repeat(x, repeats, axis=0)

        # Backward pass: define gradient
        def grad(dy):
            # For repeat_interleave, the gradient is typically accumulated back to the source indices.
            # Here we return a simple gradient for the sake of the test structure.
            return dy, None

        return y, grad

    # To execute the TensorFlow custom gradient operation, we must convert the PyTorch tensors.
    # Note: This moves data off MPS, but preserves the logic flow of the original bug report.
    # The original bug was specific to MPS slicing, but the replacement forces a TensorFlow implementation.
    data_tf = tf.convert_to_tensor(data.cpu().numpy())
    counts_slice_tf = tf.convert_to_tensor(counts[1:3].cpu().numpy())

    # Execute the replaced API
    result = custom_repeat_op(data_tf, counts_slice_tf)

    # Assertions to verify logic preservation
    # Original logic: data=[0, 1], counts_slice=[1, 0] -> result=[0]
    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    assert result[0].numpy() == 0, f"Expected value 0, got {result[0].numpy()}"

if __name__ == "__main__":
    test_issue_167924_replacement()
    print("Test passed.")