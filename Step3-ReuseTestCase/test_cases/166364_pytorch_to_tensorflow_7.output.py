import torch
import tensorflow as tf
from tensorflow.keras import backend as K

def test_name_scope_with_learnable_scalar(use_compile=False):
    """
    Test function to verify learnable scalars work within tf.keras.backend.name_scope.
    This adapts the PyTorch Flex Attention bug reproduction logic to TensorFlow.
    
    The original PyTorch bug (Issue 166364) occurred when a learnable scalar 
    (nn.Parameter) was used inside a score modification function, causing 
    vmap/compilation errors. This test verifies if the TensorFlow equivalent 
    (tf.Variable) works correctly inside the name_scope context, with and 
    without tf.function compilation.
    """
    # Equivalent to PyTorch: temp = nn.Parameter(torch.tensor(0.0))
    temp = tf.Variable(0.0, name="learnable_scalar")

    # Equivalent to PyTorch: def score_mod(score, b, h, q, kv): ...
    # We simplify the signature to match standard TF ops, focusing on the 
    # interaction between the variable and the tensor.
    def score_mod(score):
        # Use the API under test: tf.keras.backend.name_scope
        with K.name_scope("score_mod_scope"):
            # Mimic the bug logic: score = score + temp
            # In the PyTorch bug, adding a scalar to a batched tensor inside 
            # a compiled/vmap context failed.
            return score + temp

    # Create dummy input data (Batch=2, Seq=4)
    # Equivalent to the 'score' tensor in Flex Attention
    score = tf.constant([[1.0, 2.0, 3.0, 4.0], 
                         [5.0, 6.0, 7.0, 8.0]])

    if use_compile:
        # Equivalent to PyTorch: torch.compile(flex_attention)
        score_mod = tf.function(score_mod)

    # 1. Test Forward Pass
    # The PyTorch bug reported: "Without compiling: Cause error ... in backward. Forward works well."
    # "With compiling: Forward fails too."
    result = score_mod(score)
    
    # Verify output shape and values
    # Expected: score + 0.0
    expected = score
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), "Forward pass failed: output mismatch"
    print(f"Forward pass successful. Result shape: {result.shape}")

    # 2. Test Backward Pass (Gradient Check)
    # The PyTorch bug reported errors in backward pass without compilation.
    with tf.GradientTape() as tape:
        output = score_mod(score)
        loss = tf.reduce_sum(output)
    
    grads = tape.gradient(loss, temp)
    
    # Verify that the learnable scalar is tracked and gradients are computed
    assert grads is not None, "Backward pass failed: Gradient is None for learnable scalar"
    # Gradient of sum(score + temp) w.r.t temp should be batch_size * seq_len
    expected_grad = 2.0 * 4.0 
    assert tf.abs(grads - expected_grad) < 1e-5, f"Gradient mismatch: {grads} != {expected_grad}"
    print(f"Backward pass successful. Gradient: {grads.numpy()}")

if __name__ == "__main__":
    print("Testing tf.keras.backend.name_scope without compilation...")
    test_name_scope_with_learnable_scalar(use_compile=False)
    
    print("\nTesting tf.keras.backend.name_scope with compilation (tf.function)...")
    test_name_scope_with_learnable_scalar(use_compile=True)