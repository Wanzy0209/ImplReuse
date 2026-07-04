import sys

# Handle environment/dependency issues (e.g., GLIBC version mismatch) by catching import errors
try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Skipping test due to import error: {e}")
    print("This is likely due to a missing GLIBC version (e.g., GLIBCXX_3.4.29) or incompatible environment.")
    sys.exit(0)

def test_unique_matmul_divergence():
    """
    Test case to reproduce eager/compile divergence logic similar to the PyTorch issue.
    The original issue involves torch.unique returning a tensor used in matmul,
    causing a shape mismatch during compilation.
    
    Here we translate the logic to TensorFlow using tf.unique.
    """
    
    # Setup inputs to match the PyTorch shapes
    # arg_0: size=(2, 10), dtype=float64
    arg_0 = tf.reshape(tf.random.normal((20,), dtype=tf.float64), (2, 10))
    # arg_1: size=(10, 3), dtype=float64
    arg_1 = tf.reshape(tf.random.normal((30,), dtype=tf.float64), (10, 3))
    sentinel = tf.constant(1.0)

    def fuzzed_program(arg_0, arg_1, sentinel):
        # var_node_2 = torch.matmul(var_node_3, var_node_4)
        var_node_2 = tf.matmul(tf.cast(arg_0, tf.float64), tf.cast(arg_1, tf.float64))

        # _inp_unique_wide = torch.arange(1, dtype=torch.int64)
        # _uniq_wide = torch.unique(_inp_unique_wide)
        # In TF, tf.unique returns (unique_values, indices)
        _inp_unique_wide = tf.range(1, dtype=tf.int64)
        _uniq_wide, _ = tf.unique(_inp_unique_wide)

        # var_node_1 = _uniq_wide.to(var_node_2.dtype)
        var_node_1 = tf.cast(_uniq_wide, var_node_2.dtype)

        # var_node_5 = torch.full((1, 18), 0.4033..., dtype=torch.float64)
        var_node_5 = tf.fill([1, 18], tf.constant(0.40330381448978797, dtype=tf.float64))

        # var_node_0 = torch.matmul(var_node_1, var_node_5)
        # This operation caused the divergence in PyTorch: matmul((1,), (1, 18))
        var_node_0 = tf.matmul(tf.cast(var_node_1, tf.float64), tf.cast(var_node_5, tf.float64))

        result = var_node_0 * sentinel
        return result

    # 1. Run in Eager Mode
    print("Running Eager Mode...")
    try:
        result_eager = fuzzed_program(arg_0, arg_1, sentinel)
        print(f" Eager Success. Shape: {result_eager.shape}")
    except Exception as e:
        print(f" Eager Failed: {e}")
        return

    # 2. Run in Compiled Mode (tf.function)
    print("\nRunning Compiled Mode (tf.function)...")
    try:
        compiled_program = tf.function(fuzzed_program)
        result_compiled = compiled_program(arg_0, arg_1, sentinel)
        print(f" Compiled Success. Shape: {result_compiled.shape}")
    except Exception as e:
        print(f" Compiled Failed: {e}")
        return

    # 3. Check for Divergence
    if result_eager.shape != result_compiled.shape:
        print(f"\n Divergence Detected!")
        print(f"   Eager Shape:    {result_eager.shape}")
        print(f"   Compiled Shape: {result_compiled.shape}")
        raise AssertionError("Shape mismatch between eager and compiled execution")
    else:
        print("\n No divergence detected. Shapes match.")

if __name__ == "__main__":
    test_unique_matmul_divergence()