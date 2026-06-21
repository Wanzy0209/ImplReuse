import tensorflow as tf
import numpy as np

# Enable eager execution to allow using Tensors in Python control flow (assertions)
# This addresses the "OperatorNotAllowedInGraphError"
tf.compat.v1.enable_eager_execution()

# The original issue involves handling NaN values in a grid sampling operation.
# The similar API is tf.compat.v1.VariableAggregation.
# We test if variables configured with specific aggregation modes handle NaN values correctly.

def test_variable_aggregation_nan_handling():
    # Input data containing NaN, similar to grid_nan in the original issue
    input_data = [1.0, float('nan'), 3.0]

    # Test with VariableAggregation.NONE (value 0, matching the '0' argument in the original bug report)
    # and other modes to ensure consistency.
    modes = [
        tf.compat.v1.VariableAggregation.NONE,
        tf.compat.v1.VariableAggregation.SUM,
        tf.compat.v1.VariableAggregation.MEAN
    ]

    strategy = tf.distribute.get_strategy()

    with strategy.scope():
        for mode in modes:
            # Create variable with NaN and specific aggregation mode
            var = tf.Variable(input_data, aggregation=mode)
            result = var.read_value()

            # Check if NaN is preserved (The original bug had NaN -> 1.0)
            # We expect NaN to remain NaN in the variable state
            has_nan = tf.reduce_any(tf.math.is_nan(result))
            
            # Evaluate the tensor to a Python boolean for the assertion
            assert has_nan.numpy(), f"NaN should be preserved in variable with aggregation mode {mode.name}"

    print("Test passed: NaN values are handled correctly across VariableAggregation modes.")

if __name__ == "__main__":
    test_variable_aggregation_nan_handling()