import torch
import tensorflow as tf

def test_outer_with_sliced_tensor():
    """
    Adapted test case based on PyTorch Issue 167924.
    Original bug: Crash on MPS when using repeat_interleave with a sliced tensor.
    Target API: tf.keras.ops.outer
    
    This test verifies if tf.keras.ops.outer handles non-prefix sliced tensors
    correctly without crashing, preserving the input structure of the original bug.
    """
    # Setup inputs mimicking the original PyTorch code
    # Original: data = torch.arange(2, device="mps")
    a = tf.range(2)
    
    # Original: counts = torch.tensor([0, 1, 0], device="mps")
    b = tf.constant([0, 1, 0])
    
    # Original: counts[1:3] (Non-prefix slice)
    # The core bug reproduction logic involves passing a sliced tensor.
    b_sliced = b[1:3]
    
    # Original: data.repeat_interleave(counts[1:3], dim=0)
    # Adaptation: Call the similar API (tf.keras.ops.outer) with the sliced tensor.
    # We check if the operation completes successfully and produces the correct result.
    result = tf.keras.ops.outer(a, b_sliced)
    
    # Verify the result
    # a = [0, 1], b_sliced = [1, 0]
    # Expected outer product: [[0*1, 0*0], [1*1, 1*0]] = [[0, 0], [1, 1]]
    expected = tf.constant([[0, 0], [1, 1]])
    
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), \
        f"Expected {expected.numpy()}, but got {result.numpy()}"
    
    print("Test passed: tf.keras.ops.outer handled sliced tensor correctly.")

if __name__ == "__main__":
    test_outer_with_sliced_tensor()