import torch
import tensorflow as tf
import numpy as np

# 1. Use the target API: Enable eager execution
# This must be called at program startup before any other TensorFlow operations.
tf.compat.v1.enable_eager_execution()

def test_attention_eager_backward():
    """
    Adapted test case for tf.compat.v1.enable_eager_execution.
    
    Original Bug Context: FlexAttention backward compilation failure with GQA on B200.
    The original bug occurred in 'inductor' (compiled) mode, while 'eager' mode worked.
    This test verifies that the equivalent attention operation runs successfully
    in TensorFlow's eager execution mode with gradient computation.
    """
    
    # 2. Setup inputs
    # Original PyTorch shapes: [Batch, Heads, Seq, Dim]
    # PyTorch Q: [2, 32, 4096, 128], K/V: [2, 8, 4096, 128] (GQA configuration)
    # TensorFlow Keras layers typically expect [Batch, Seq, Features].
    # We adapt the dimensions to fit a standard MultiHeadAttention layer.
    
    batch_size = 2
    seq_len = 4096
    num_heads = 32
    head_dim = 128
    
    # Total feature dimension for the input tensors
    feature_dim = num_heads * head_dim
    
    # Initialize tensors with random values
    # Using float32 to ensure gradient stability in this test context
    q = tf.random.normal([batch_size, seq_len, feature_dim])
    k = tf.random.normal([batch_size, seq_len, feature_dim])
    v = tf.random.normal([batch_size, seq_len, feature_dim])

    # 3. Define the Model
    # Using tf.keras.layers.MultiHeadAttention as the closest equivalent to flex_attention.
    # Note: Standard TF MHA does not natively support GQA (mismatched KV heads) 
    # in the same way the PyTorch flex_attention call does without custom logic.
    # We test the standard MHA path to verify eager execution stability.
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_dim
    )

    # 4. Run Forward Pass
    # We use tf.GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch([q, k, v])
        
        # Perform attention
        # In TF, query is the first arg, value and key follow.
        y = attention_layer(q, v, k)
        
        # Create a loss to backpropagate (mimicking y.backward(torch.randn_like(y)))
        loss = tf.reduce_sum(y)

    # 5. Run Backward Pass
    grads = tape.gradient(loss, [q, k, v])

    # 6. Assertions
    # Verify that gradients were computed successfully (None would imply a break in the graph)
    assert grads[0] is not None, "Gradient for Q is None"
    assert grads[1] is not None, "Gradient for K is None"
    assert grads[2] is not None, "Gradient for V is None"
    
    # Verify shapes match inputs
    assert grads[0].shape == q.shape
    assert grads[1].shape == k.shape
    assert grads[2].shape == v.shape

    print("Test passed: Eager execution handles attention forward and backward passes successfully.")

if __name__ == "__main__":
    test_attention_eager_backward()