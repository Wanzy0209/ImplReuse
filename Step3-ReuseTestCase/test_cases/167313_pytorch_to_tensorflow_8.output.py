import torch
import tensorflow as tf
import numpy as np

def test_tf_addmm_alpha_beta():
    """
    Adapts the PyTorch addmm bug reproduction to TensorFlow.
    The original bug involves torch.compile ignoring alpha/beta parameters
    when optimizing addmm -> add(mm).
    
    Here we test the TensorFlow equivalent logic using tf.matmul and tf.add
    (or tf.raw_ops.Addmm) under tf.function(jit_compile=True) to check if
    the XLA compiler correctly preserves scaling factors.
    
    We use tf.compat.v1.convert_to_tensor as requested for input creation.
    """
    
    # Enable/disable v2 behavior to ensure compat.v1 works if needed, 
    # though convert_to_tensor usually works fine in TF2.
    tf.compat.v1.disable_eager_execution() 
    # Note: Disabling eager execution makes testing harder without sessions.
    # Let's stick to TF2 eager + tf.function which is the standard way now.
    # tf.compat.v1.convert_to_tensor is available in eager mode.
    
    # 1. Setup inputs using the specified API
    # x = torch.rand(2, device="cuda")
    np_x = np.random.rand(2).astype(np.float32)
    x = tf.compat.v1.convert_to_tensor(np_x, dtype=tf.float32)
    
    # a = torch.rand(2, 3, device="cuda")
    np_a = np.random.rand(2, 3).astype(np.float32)
    a = tf.compat.v1.convert_to_tensor(np_a, dtype=tf.float32)
    
    # b = torch.rand(3, 2, device="cuda")
    np_b = np.random.rand(3, 2).astype(np.float32)
    b = tf.compat.v1.convert_to_tensor(np_b, dtype=tf.float32)

    alpha = 0.5
    beta = 0.5

    # 2. Define the operation logic
    # PyTorch: torch.addmm(x, a, b, alpha=0.5, beta=0.5) -> beta * x + alpha * (a @ b)
    # Note: addmm broadcasts x to the shape of the matrix multiplication result.
    
    def addmm_logic(x, a, b):
        mm = tf.matmul(a, b)
        # Broadcasting x (shape [2]) to mm (shape [2, 2])
        # In numpy/TF, [2] + [2, 2] broadcasts the [2] to [1, 2] then to [2, 2]
        return beta * x + alpha * mm

    # 3. Eager execution (Baseline)
    result_eager = addmm_logic(x, a, b)
    
    # 4. Compiled execution (Simulating torch.compile)
    # Using jit_compile=True to invoke XLA, similar to Inductor
    @tf.function(jit_compile=True)
    def compiled_addmm(x, a, b):
        return addmm_logic(x, a, b)
        
    result_compiled = compiled_addmm(x, a, b)

    # 5. Verification
    # Check if the results are close. 
    # If the bug existed in TF (ignoring alpha/beta), they would differ significantly.
    print("Eager Result:\n", result_eager.numpy())
    print("Compiled Result:\n", result_compiled.numpy())
    
    # Assert that the compiled version respects alpha and beta
    # We calculate the expected value manually to be sure
    expected_mm = np.matmul(np_a, np_b)
    expected_result = beta * np_x + alpha * expected_mm
    
    np.testing.assert_allclose(result_eager.numpy(), expected_result, rtol=1e-5)
    np.testing.assert_allclose(result_compiled.numpy(), expected_result, rtol=1e-5)
    
    print("Test Passed: TensorFlow respects alpha/beta in both eager and compiled modes.")

if __name__ == "__main__":
    test_tf_addmm_alpha_beta()