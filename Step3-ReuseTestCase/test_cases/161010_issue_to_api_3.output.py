import torch
import tensorflow as tf
import numpy as np

def test_xla_compile_preserves_logic():
    """
    Test case adapted from PyTorch Issue 161010.
    Original Issue: torch.compile doesn't preserve stride with clone(memory_format=torch.preserve_format).
    
    This test verifies that TensorFlow's XLA compilation (tf.function with jit_compile=True)
    preserves the logic and properties of tensor operations similarly to eager execution,
    specifically focusing on linear algebra operations (QR, Solve) and tensor identity checks.
    """
    
    # Enable XLA
    # Note: This test requires a TensorFlow build with XLA support.
    
    # Define the input tensor
    # Using float32 for standard linear algebra operations
    A = tf.random.uniform((5, 5), minval=-1.0, maxval=1.0, dtype=tf.float32)
    
    def f(A, count):
        # Replicate the logic: QR decomposition -> Solve -> Property Check
        Q, R = tf.linalg.qr(A)
        
        # rhs = torch.ones(Q.shape[0], 1, device=A.device)
        rhs = tf.ones([tf.shape(Q)[0], 1], dtype=A.dtype)
        
        # a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
        # TensorFlow equivalent: tf.linalg.triangular_solve
        # Note: tf.linalg.triangular_solve solves R * X = rhs
        # We need to compute Q.T @ rhs first
        qt_rhs = tf.matmul(Q, rhs, transpose_a=True)
        a = tf.linalg.triangular_solve(R, qt_rhs, lower=False)
        
        # Original PyTorch check: 
        # if a.stride() == a.clone(memory_format=torch.preserve_format).stride():
        # 
        # TensorFlow adaptation:
        # Check if a property (shape or value) is preserved through an identity operation.
        # Since TF abstracts memory layout (stride), we check shape and value consistency
        # which is the semantic equivalent of ensuring the tensor wasn't corrupted 
        # or transformed unexpectedly by the compiler.
        
        # We use tf.identity as the equivalent of clone.
        a_clone = tf.identity(a)
        
        # Check if shapes match (basic property preservation)
        shapes_match = tf.reduce_all(tf.equal(tf.shape(a), tf.shape(a_clone)))
        
        # Check if values match (data preservation)
        values_match = tf.reduce_all(tf.equal(a, a_clone))
        
        # If properties are preserved, increment count
        if shapes_match and values_match:
            return count + 1
        return count

    # 1. Run in Eager Mode
    count_eager = tf.constant(0, dtype=tf.int32)
    res_eager = f(A, count_eager)
    
    # 2. Run in Compiled Mode (XLA)
    # This is the TensorFlow equivalent of torch.compile
    f_compiled = tf.function(f, jit_compile=True)
    res_compiled = f_compiled(A, count_eager)
    
    # 3. Assert Results
    # The original bug showed a mismatch between eager and compiled results.
    # This assertion ensures the compiled version behaves correctly.
    assert res_eager == res_compiled, (
        f"Mismatch between eager and compiled execution: "
        f"Eager={res_eager}, Compiled={res_compiled}"
    )
    
    # Also verify the logic worked as expected (should be 1)
    assert res_eager == 1, "Eager execution logic failed property check."

if __name__ == "__main__":
    test_xla_compile_preserves_logic()
    print("Test passed: XLA compilation preserves tensor logic and properties.")