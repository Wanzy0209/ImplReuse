import tensorflow as tf

# Enable eager execution as requested by the target API
tf.compat.v1.enable_eager_execution()

def test_attention_bias_grad():
    # Define dimensions
    B, L, D = 2, 16, 64

    # Force execution on CPU to avoid cuBLAS incompatibility issues
    # The error indicates a failure in the GPU BLAS library (CUBLAS_STATUS_EXECUTION_FAILED),
    # likely due to running an older TensorFlow/CUDA version (10.0) on newer hardware (RTX 4090).
    # Moving to CPU ensures the test logic (gradient flow) can be verified without hardware crashes.
    with tf.device('/CPU:0'):
        # Create inputs with gradient tracking
        x = tf.random.normal((B, L, D))
        y = tf.random.normal((B, L))

        with tf.GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)

            # Materialize a bias matrix
            # Mimicking the logic: bias_mat[b, q_idx] + y[b, kv_idx]
            y_expanded_q = tf.expand_dims(y, axis=2)  # (B, L, 1)
            y_expanded_kv = tf.expand_dims(y, axis=1) # (B, 1, L)
            bias_mat = y_expanded_q + y_expanded_kv  # (B, L, L)

            # Simplified Attention Logic (Manual implementation of flex_attention equivalent)
            q = x
            k = x
            v = x

            # Calculate scores (Q * K^T)
            # This operation was causing the Blas xGEMMBatched launch failure on GPU
            scores = tf.matmul(q, k, transpose_b=True) # (B, L, L)

            # Apply score_mod (add bias)
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