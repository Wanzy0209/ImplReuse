import torch
import tensorflow as tf
import numpy as np

def test_isrealobj_compilation():
    """
    Adapted test case for tf.experimental.numpy.isrealobj based on the 
    torch.addmm bug report pattern (Issue 167313).
    
    The original bug involved parameters (alpha/beta) being ignored during 
    compilation (torch.compile). While isrealobj does not have alpha/beta 
    parameters, this test verifies that the API behaves consistently 
    between eager execution and tf.function (compiled) execution, ensuring 
    no logic is lost during the compilation process.
    """
    
    # Setup inputs relevant to isrealobj
    # 1. A real-valued tensor
    x_real = tf.constant([1.0, 2.0, 3.0])
    
    # 2. A complex-valued tensor
    x_complex = tf.constant([1.0 + 2.0j, 3.0 + 4.0j])

    # Define the function using the target API
    # Original: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))
    # Adapted: Check if the object is real
    f = lambda x: tf.experimental.numpy.isrealobj(x)

    # Compile the function (TensorFlow equivalent of torch.compile)
    fc = tf.function(f)

    # --- Test Real Input ---
    # Eager execution
    res_eager_real = f(x_real)
    # Compiled execution
    res_compiled_real = fc(x_real)

    print(f"Real Input - Eager: {res_eager_real}")
    print(f"Real Input - Compiled: {res_compiled_real}")

    # --- Test Complex Input ---
    # Eager execution
    res_eager_complex = f(x_complex)
    # Compiled execution
    res_compiled_complex = fc(x_complex)

    print(f"Complex Input - Eager: {res_eager_complex}")
    print(f"Complex Input - Compiled: {res_compiled_complex}")

    # Assertions
    # Verify correctness for real input
    assert res_eager_real == True, "Eager: Real object should return True"
    assert res_compiled_real == True, "Compiled: Real object should return True"
    
    # Verify correctness for complex input
    assert res_eager_complex == False, "Eager: Complex object should return False"
    assert res_compiled_complex == False, "Compiled: Complex object should return False"

    # Verify consistency between eager and compiled modes
    # This mirrors the original bug report where eager and compiled results differed
    assert res_eager_real == res_compiled_real, "Mismatch: Real input results differ between eager and compiled"
    assert res_eager_complex == res_compiled_complex, "Mismatch: Complex input results differ between eager and compiled"

if __name__ == "__main__":
    test_isrealobj_compilation()
    print("Test passed.")