import torch
import tensorflow as tf

def get_score_function():
    """
    Constructs a function equivalent to the PyTorch example:
    vmap(grad(logp)).
    Wrapped in tf.compat.v1.name_scope to test the API.
    """

    def logp(x, matrix):
        # Using tf.compat.v1.name_scope as the target API
        with tf.compat.v1.name_scope("logp_computation"):
            # PyTorch: torch.linalg.cholesky(matrix).contiguous()
            # TensorFlow: tf.linalg.cholesky
            p_mat_sqrt = tf.linalg.cholesky(matrix)

            # PyTorch: p_mat_sqrt.inverse()
            # TensorFlow: tf.linalg.inv
            p_mat_sqrt_inv = tf.linalg.inv(p_mat_sqrt)

            # PyTorch: torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
            # Note: x is (5, 3), x[0, :] is (3,). p_mat_sqrt_inv is (3, 3).
            # We use matvec for matrix-vector multiplication.
            val = tf.reduce_sum(tf.square(tf.linalg.matvec(p_mat_sqrt_inv, x[0, :])))

            return -val / 2.0

    @tf.function
    def score_func(x_batch, matrix):
        with tf.compat.v1.name_scope("vmap_grad_scope"):
            # PyTorch: torch.func.grad(logp, 0)
            def grad_fn(x_single):
                with tf.GradientTape() as tape:
                    tape.watch(x_single)
                    loss = logp(x_single, matrix)
                return tape.gradient(loss, x_single)

            # PyTorch: torch.vmap(..., (0, None))
            # TensorFlow: tf.vectorized_map
            return tf.vectorized_map(grad_fn, x_batch)

    return score_func

if __name__ == "__main__":
    # Data setup matching PyTorch: (2, 5, 3), float32
    data = tf.zeros((2, 5, 3), dtype=tf.float32)

    # Matrix setup: diag([20, 0.5, 5]^2)
    diag_vals = tf.constant([20., 0.5, 5.], dtype=tf.float32) ** 2
    p = tf.linalg.diag(diag_vals)

    # Get the function
    compiled_function = get_score_function()

    # Execute
    try:
        res = compiled_function(data, p)
        print("Test passed. Result shape:", res.shape)
        # Verify result is zero (gradient at zero)
        assert tf.reduce_all(tf.equal(res, 0.0))
        print("Result values verified (all zeros).")
    except Exception as e:
        print(f"Test failed with error: {e}")