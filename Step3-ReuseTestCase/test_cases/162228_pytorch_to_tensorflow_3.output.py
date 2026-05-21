import tensorflow as tf

# Enable eager execution as requested by the target API
tf.compat.v1.enable_eager_execution()

def test_attention_bias_grad():
    # Define dimensions
    B, L, D = 2, 16, 64

    # Create inputs with gradient tracking
    # Note: In TF eager, we use tf.Variable for trainable parameters or watch tensors in tape.
    # Here we use tensors and watch them to mimic the requires_grad=True behavior.
    x = tf.random.normal((B, L, D))
    y = tf.random.normal((B, L))

    with tf.GradientTape() as tape:
        tape.watch(x)
        tape.watch(y)

        # Materialize a bias matrix
        # Mimicking the logic: bias_mat[b, q_idx] + y[b, kv_idx]
        # y shape is (B, L). We want bias_mat shape (B, L, L).
        # y[b, q_idx] -> (B, L, 1)
        # y[b, kv_idx] -> (B, 1, L)
        y_expanded_q = tf.expand_dims(y, axis=2)  # (B, L, 1)
        y_expanded_kv = tf.expand_dims(y, axis=1) # (B, 1, L)
        bias_mat = y_expanded_q + y_expanded_kv  # (B, L, L)

        # Simplified Attention Logic (Manual implementation of flex_attention equivalent)
        # Q, K, V from x
        # PyTorch code repeats x to (B, L, 16, 1) for heads, but for minimal TF test 
        # we stick to standard (B, L, D) matrix multiplication.
        q = x
        k = x
        v = x

        # Calculate scores (Q * K^T)
        scores = tf.matmul(q, k, transpose_b=True) # (B, L, L)

        # Apply score_mod (add bias)
        # This corresponds to the flex_attention score_mod argument
        scores_mod = scores + bias_mat

        # Softmax
        attn_weights = tf.nn.softmax(scores_mod, axis=-1)

        # Output
        out = tf.matmul(attn_weights, v)

        # Loss
        loss = tf.reduce_mean(out)

    # Compute gradients
    grads = tape.gradient(loss, [x, y])

    # Verify gradients
    grad_x, grad_y = grads
    
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"x: {(grad_x is not None) and (tf.norm(grad_x) > 0)}, y: {(grad_y is not None) and (tf.norm(grad_y) > 0)}")
    
    assert grad_x is not None, "Gradient for x is None"
    assert tf.norm(grad_x) > 0, "Gradient for x is zero"
    assert grad_y is not None, "Gradient for y is None"
    assert tf.norm(grad_y) > 0, "Gradient for y is zero"

if __name__ == "__main__":
    test_attention_bias_grad()