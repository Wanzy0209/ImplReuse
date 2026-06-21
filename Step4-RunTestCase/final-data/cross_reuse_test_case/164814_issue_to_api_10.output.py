import torch
import os
import sys

# Attempt to import TensorFlow, handling potential environment incompatibilities.
try:
    import tensorflow as tf
except ImportError as e:
    print("Skipping test: Failed to import TensorFlow.")
    print("This is likely due to a missing system dependency (e.g., GLIBCXX_3.4.29 not found).")
    print(f"Error details: {e}")
    sys.exit(0)

# Ensure the environment is set up for the similar API behavior.
# The bug report involves operations on a scalar (0-d tensor).
# tf.experimental.dtensor.num_clients returns 1 in local mode (scalar count),
# which we will use to construct a tensor that gets squeezed to a scalar.
if "DTENSOR_JOBS" in os.environ:
    del os.environ["DTENSOR_JOBS"]

def test_dtensor_num_clients_dimensionality():
    """
    Test case inspired by Issue 164814.
    
    The original bug involves a divergence between eager and compiled modes
    when handling tensor dimensionality (specifically squeezing a size-1 tensor
    to a scalar). This test adapts that logic using tf.experimental.dtensor.num_clients.
    
    We use num_clients() (which returns 1 in local mode) to define a tensor shape,
    then perform reshape/squeeze operations to verify consistency in eager vs graph modes.
    """
    
    # Sentinel variable to mimic the gradient/dependency logic in the original bug
    sentinel = tf.Variable(1.0, dtype=tf.float32)

    def logic_fn():
        # Leverage the similar API: tf.experimental.dtensor.num_clients
        # In local mode (default here), this returns 1.
        n_clients = tf.experimental.dtensor.num_clients()
        
        # Create a tensor based on the API return value.
        # Mimics: var_node_2 = ... # size=(1,)
        # We use int32 to match the original bug's dtype where possible
        t = tf.ones([n_clients], dtype=tf.int32)
        
        # Mimics: var_node_1 = torch.reshape(var_node_2, [1])
        t = tf.reshape(t, [n_clients])
        
        # Mimics: var_node_0 = torch.squeeze(var_node_1) # size=()
        # If n_clients is 1, this results in a scalar (0-d tensor).
        # This is the operation that triggered the dimensionality error in PyTorch.
        t = tf.squeeze(t)
        
        # Mimics: result = var_node_0 * sentinel
        # Ensure the scalar interacts with other tensors.
        result = tf.cast(t, tf.float32) * sentinel
        return result

    # 1. Run in Eager mode
    eager_result = logic_fn()
    
    # 2. Run in Compiled mode (tf.function is analogous to torch.compile)
    compiled_fn = tf.function(logic_fn)
    compiled_result = compiled_fn()

    # 3. Assert results match (checking for divergence)
    # The original bug reported a crash/error in compiled mode.
    # Here we verify that the outputs are identical.
    assert tf.equal(eager_result, compiled_result).numpy(), \
        f"Divergence detected: Eager={eager_result.numpy()}, Compiled={compiled_result.numpy()}"
    
    # Additionally, verify the shape logic holds (scalar result)
    assert eager_result.shape == (), f"Expected scalar result, got {eager_result.shape}"
    assert compiled_result.shape == (), f"Expected scalar result in compiled mode, got {compiled_result.shape}"

    print(" Test passed: No divergence between eager and compiled modes with scalar outputs.")

if __name__ == "__main__":
    test_dtensor_num_clients_dimensionality()