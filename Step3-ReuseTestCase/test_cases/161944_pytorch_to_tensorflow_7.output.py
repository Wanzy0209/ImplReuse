import torch
import tensorflow as tf
import numpy as np

# Adapted from PyTorch test case for torch.exp
# Original Issue: Inductor (torch.compile) uses fast math for exp, causing precision loss.
# Adaptation: Check if tf.function (TensorFlow's compilation/graph mode) 
# alters the behavior of operations involving tf.VariableAggregation 
# compared to eager execution, using float64 as a reference.

def test_variable_aggregation_precision():
    # Setup input data
    inp = tf.random.normal((8192,))
    
    # Define the aggregation type to test
    # Note: VariableAggregation is primarily used in distributed contexts,
    # but we test its basic usage here to mirror the API usage.
    aggregation_type = tf.VariableAggregation.SUM

    # 1. Eager execution (float32)
    # We simulate a reduction operation using the aggregation API
    v_eager = tf.Variable(0.0, dtype=tf.float32, aggregation=aggregation_type)
    v_eager.assign_add(tf.reduce_sum(inp))
    out1 = v_eager.read_value()

    # 2. Compiled execution (float32) - equivalent to torch.compile
    @tf.function
    def compiled_step(var, data):
        var.assign_add(tf.reduce_sum(data))
        return var.read_value()

    v_compiled = tf.Variable(0.0, dtype=tf.float32, aggregation=aggregation_type)
    out2 = compiled_step(v_compiled, inp)

    # 3. High precision reference (float64)
    v_high = tf.Variable(0.0, dtype=tf.float64, aggregation=aggregation_type)
    v_high.assign_add(tf.reduce_sum(tf.cast(inp, tf.float64)))
    out3_high = v_high.read_value()

    # Calculate differences
    diff_eager = tf.abs(out3_high - tf.cast(out1, tf.float64)).numpy()
    diff_compiled = tf.abs(out3_high - tf.cast(out2, tf.float64)).numpy()

    print(f"Max diff (Eager vs High): {diff_eager}")
    print(f"Max diff (Compiled vs High): {diff_compiled}")

    # Basic assertion to ensure compiled mode doesn't deviate wildly
    # (In the original bug, compiled mode had significantly higher error)
    assert diff_compiled < 1e-5, "Compiled mode precision deviation detected"

if __name__ == "__main__":
    test_variable_aggregation_precision()