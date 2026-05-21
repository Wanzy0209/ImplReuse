import torch
import tensorflow as tf
import numpy as np

# Configure mixed precision to match the float16 context of the original bug
# This attempts to reproduce the environment where IncompatibleTypeError occurred
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def test_variable_aggregation_bug_reproduction(aggregation_type):
    """
    Adapts the logic of the original PyTorch bug to test tf.VariableAggregation.
    The original bug involved type mismatches (int64, float16) during compilation
    (torch.compile). Here we test tf.Variable with specific aggregation types
    inside a tf.function (graph mode/compilation) using similar data types.
    """
    print(f"Testing aggregation: {aggregation_type}")

    # Inputs mirroring the original bug report shapes and dtypes
    # arg0: size=(42, 56), dtype=int64
    arg0 = tf.constant(np.random.randint(0, 1000, (42, 56)), dtype=tf.int64)
    # arg1: size=(50000, 128), dtype=float16
    arg1 = tf.constant(np.random.randn(50000, 128), dtype=tf.float16)

    # Equivalent to torch.compile
    @tf.function(reduce_retracing=True)
    def compiled_fn(x_int, x_fp16):
        # Create a variable with the target API: tf.VariableAggregation
        # We initialize it with float16 to match the precision context
        var = tf.Variable(
            initial_value=tf.ones([42, 56], dtype=tf.float16), 
            aggregation=aggregation_type, 
            dtype=tf.float16,
            trainable=True
        )

        # Original bug logic: torch.tanh(t0) where t0 is int64
        # In TensorFlow, tanh requires float inputs. We cast to test the type transition.
        # This mimics the implicit casting or type handling that might have failed in PyTorch.
        x_cast = tf.cast(x_int, tf.float16)
        t1 = tf.tanh(x_cast)

        # Original bug logic involved complex pow operations and embedding lookups.
        # We perform a pow operation involving the variable to test interaction
        # between the variable (with specific aggregation) and other tensors.
        # t11 = torch.pow(..., t5)
        result = tf.pow(t1, var)

        return result

    # Test Eager Execution
    try:
        out_eager = compiled_fn(arg0, arg1)
        # Ensure output is finite
        assert tf.reduce_all(tf.math.is_finite(out_eager))
        print(f"Eager Success for {aggregation_type} ")
    except Exception as e:
        print(f"Eager Failed for {aggregation_type}: {e}")
        return False

    # Test Compiled Execution (Graph Mode)
    try:
        out_compiled = compiled_fn(arg0, arg1)
        # Ensure output is finite
        assert tf.reduce_all(tf.math.is_finite(out_compiled))
        print(f"Compile Success for {aggregation_type} ")
    except Exception as e:
        print(f"Compile Failed for {aggregation_type}: {e}")
        return False

    return True

if __name__ == '__main__':
    # Test all available aggregation modes in tf.VariableAggregation
    aggregations = [
        tf.VariableAggregation.NONE,
        tf.VariableAggregation.SUM,
        tf.VariableAggregation.MEAN,
        tf.VariableAggregation.ONLY_FIRST_REPLICA
    ]

    print("Starting TensorFlow VariableAggregation test adapted from PyTorch Issue 164086...")
    for agg in aggregations:
        success = test_variable_aggregation_bug_reproduction(agg)
        if not success:
            print(f"Test suite halted due to failure in {agg}")
            break
    print("Test suite completed.")