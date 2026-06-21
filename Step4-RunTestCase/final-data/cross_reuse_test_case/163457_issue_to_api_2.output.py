import torch
import tensorflow as tf
import numpy as np

# The original issue (Issue ID: 163457) reports a "divmod" error involving 'SymInt' and 'int'
# during PyTorch compilation (torch.compile). This suggests a type mismatch in arithmetic
# operations (specifically division/modulo) on symbolic values.
#
# The similar API, tf.experimental.numpy.geomspace, contains a floor division operation
# ('// 2') in its implementation logic for calculating 'signflip':
# signflip = 1 - start_sign * stop_sign // 2
#
# This test case verifies that tf.experimental.numpy.geomspace executes correctly
# within a compiled context (tf.function), ensuring that the internal floor division
# does not cause type errors analogous to the PyTorch bug.

def test_geomspace_divmod_compilation():
    # Ensure numpy behavior is enabled for the experimental namespace
    tf.experimental.numpy.enable_numpy_behavior()

    # Define the function to be compiled, analogous to torch.compile
    @tf.function
    def compiled_geomspace(start, stop, num):
        return tf.experimental.numpy.geomspace(start, stop, num=num)

    # Inputs chosen to trigger the signflip logic path in geomspace
    # which contains the '//' operation.
    # Using negative values ensures the sign calculation logic is active.
    start_val = -16.0
    stop_val = -1.0
    num_val = 10

    print(f"Testing tf.experimental.numpy.geomspace with start={start_val}, stop={stop_val}, num={num_val}")

    # 1. Eager Execution (Baseline)
    try:
        out_eager = tf.experimental.numpy.geomspace(start_val, stop_val, num=num_val)
        print("Eager execution successful.")
    except Exception as e:
        print(f"Eager execution failed: {e}")
        raise

    # 2. Compiled Execution (Analogous to torch.compile in the bug report)
    # This checks if the internal '//' operation handles symbolic tensors correctly.
    try:
        out_compiled = compiled_geomspace(start_val, stop_val, num=num_val)
        print("Compiled execution successful.")
    except Exception as e:
        print(f"Compiled execution failed: {e}")
        # If this fails with a type error similar to "unsupported operand type(s) for divmod",
        # it mirrors the original bug.
        raise

    # 3. Consistency Check
    # Verify that the compiled output matches the eager output
    if not np.allclose(out_eager.numpy(), out_compiled.numpy()):
        print("Divergence detected between eager and compiled outputs!")
        print(f"Eager sum: {tf.reduce_sum(out_eager).numpy()}")
        print(f"Compiled sum: {tf.reduce_sum(out_compiled).numpy()}")
        raise AssertionError("Eager and compiled outputs differ significantly")
    else:
        print("Outputs match. Test passed.")

if __name__ == "__main__":
    test_geomspace_divmod_compilation()