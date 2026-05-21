import torch
import tensorflow as tf
import numpy as np

def example_function():
    # Define the log-likelihood function
    def logp(x, matrix):
        # Original bug note: print(matrix.is_contiguous()) # Uncomment this line to make the code run
        # In TensorFlow, we can use tf.print to inspect the tensor during graph execution,
        # or standard print for eager execution.
        # tf.print("Matrix is_contiguous check:", tf.debugging.assert_all_finite(matrix, "Matrix check"))
        
        # PyTorch: p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
        # TensorFlow: tf.linalg.cholesky
        p_mat_sqrt = tf.linalg.cholesky(matrix)
        
        # PyTorch: p_mat_sqrt_inv = p_mat_sqrt.inverse()
        # TensorFlow: tf.linalg.inv
        p_mat_sqrt_inv = tf.linalg.inv(p_mat_sqrt)
        
        # PyTorch: val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
        # TensorFlow: x[0, :] selects the first row. 
        # Note: x shape inside the map is (5, 3), so x[0] is (3,).
        # We use tf.linalg.matvec for matrix-vector multiplication or the @ operator.
        val = tf.reduce_sum(tf.square(p_mat_sqrt_inv @ x[0, :]))
        
        return -val / 2

    # PyTorch: score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))
    # TensorFlow equivalent: tf.vectorized_map with tf.GradientTape
    def score_func(x, matrix):
        def grad_fn(xi):
            with tf.GradientTape() as tape:
                tape.watch(xi)
                y = logp(xi, matrix)
            return tape.gradient(y, xi)
        
        return tf.vectorized_map(grad_fn, x)

    return score_func

if __name__ == "__main__":
    # Setup device (MPS on Mac is usually mapped to GPU in TensorFlow)
    # Fallback to CPU if GPU is not available to ensure the test is runnable.
    device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device_name):
        dtype = tf.float32
        # PyTorch: data = torch.zeros((2, 5, 3), device=device, dtype=dtype)
        data = tf.zeros((2, 5, 3), dtype=dtype)

        # Get the function to be executed
        func_to_run = example_function()

        # Original API: torch.compile(example_function())
        # Similar API: tf.keras.backend.name_scope
        # We wrap the execution in the name_scope as requested by the prompt.
        # This context manager handles naming for operations, which can differ between 
        # graph and eager modes, similar to how torch.compile changes execution modes.
        with tf.keras.backend.name_scope("compiled_function_scope"):
            # PyTorch: p = torch.diag(torch.tensor((20., 0.5, 5,), device=device, dtype=dtype)**2)
            p_diag_values = tf.constant([20., 0.5, 5.], dtype=dtype) ** 2
            p = tf.linalg.diag(p_diag_values)

            # PyTorch: res = compiled_function(data, p)
            res = func_to_run(data, p)

            # Verify the result shape and basic properties
            print("Result shape:", res.shape)
            assert res.shape == (2, 5, 3), f"Expected shape (2, 5, 3), got {res.shape}"
            
            # Check if result contains finite values (basic sanity check)
            assert tf.reduce_all(tf.math.is_finite(res)), "Result contains NaN or Inf"
            
            print("Test passed successfully.")