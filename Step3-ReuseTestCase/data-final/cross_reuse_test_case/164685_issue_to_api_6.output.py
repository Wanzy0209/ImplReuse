import torch
import tensorflow as tf
import numpy as np

def test_truncated_normal_compile_divergence():
    """
    Test case adapted from PyTorch Issue 164685.
    
    Original Issue: Divergence between eager and compiled mode in PyTorch 
    involving scalar arithmetic and type casting (int32/int64).
    
    Adaptation: Replaces the random generation logic (torch.randn) with 
    tf.raw_ops.TruncatedNormal to test if TensorFlow exhibits similar 
    eager vs. tf.function divergence under the same arithmetic stress.
    """
    
    # Set seed for reproducibility
    tf.random.set_seed(19989)

    # Define the program logic mirroring the PyTorch fuzzed_program
    def program_logic():
        # Original: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()
        # Adaptation: Use tf.raw_ops.TruncatedNormal to generate the base value
        # TruncatedNormal generates floats, we cast to int32 to match the bug's type stress.
        raw_val = tf.raw_ops.TruncatedNormal(shape=(), dtype=tf.float32, seed=19989)
        var_node_3 = tf.cast(raw_val, tf.int32) # dtype=int32

        # Original: var_node_2 = -6; var_node_1 = var_node_2 * var_node_3
        var_node_2 = -6
        var_node_1 = var_node_2 * var_node_3 # dtype=int32

        # Original: var_node_5 = torch.full((), 1, dtype=torch.int64)
        var_node_5 = tf.constant(1, dtype=tf.int64) # dtype=int64

        # Original: var_node_0 = var_node_1 / var_node_4
        # Note: The bug report involved division between int32 and int64.
        # In both PyTorch and TensorFlow, this results in a float.
        var_node_0 = var_node_1 / var_node_5

        # Original: result = var_node_0 * sentinel
        # We use a dummy tensor to mimic the sentinel multiplication logic
        sentinel = tf.constant(1.0)
        result = var_node_0 * sentinel
        
        return result

    # 1. Run in Eager mode
    try:
        result_eager = program_logic()
        print(f" Eager success: {result_eager.numpy()}")
    except Exception as e:
        print(f" Eager failure: {e}")
        raise

    # 2. Run in Compiled mode (tf.function equivalent to torch.compile)
    # Using fullgraph-like behavior (autograph)
    compiled_program = tf.function(program_logic, jit_compile=True)
    
    try:
        result_compiled = compiled_program()
        print(f" Compile success: {result_compiled.numpy()}")
    except Exception as e:
        print(f" Compile failure: {e}")
        raise

    # 3. Check for Divergence
    # The original bug reported a KeyError (crash), but the title mentions divergence.
    # We check if the values match.
    if not np.allclose(result_eager.numpy(), result_compiled.numpy()):
        raise AssertionError(
            f"Divergence detected!\n"
            f"Eager: {result_eager.numpy()}\n"
            f"Compiled: {result_compiled.numpy()}"
        )
    
    print(" Test passed: No divergence between eager and compiled execution.")

if __name__ == "__main__":
    test_truncated_normal_compile_divergence()