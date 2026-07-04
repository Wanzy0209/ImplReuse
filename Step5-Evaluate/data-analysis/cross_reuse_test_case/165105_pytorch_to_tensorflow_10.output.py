import torch
import numpy as np
import sys

# Attempt to import TensorFlow, handle environment errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment issues.")
    print(f"Error details: {e}")
    print("This is likely due to a GLIBC version mismatch (libstdc++.so.6).")
    sys.exit(0)

# Replicate the seed and configuration from the original PyTorch bug report
# to ensure deterministic behavior and similar context.
np.random.seed(70609)
tf.random.set_seed(70609)

# The original bug report highlighted a divergence between Eager and Compiled modes
# involving float16 tensors and specific shapes (14, 416).
# We adapt this to test tf.keras.metrics.sparse_categorical_accuracy.

# Shapes derived from the PyTorch fuzzer output:
# var_node_5: (14, 416) -> Used as y_pred (logits/probabilities)
# var_node_4: (14,)     -> Used as y_true (labels) shape reference
batch_size = 14
num_classes = 416

def test_sparse_categorical_accuracy_divergence():
    # 1. Setup Inputs
    # y_true: Integer ground truth values. Shape (14,)
    # We use int32 as standard for labels, but the shape matches the PyTorch vector.
    y_true = tf.constant(np.random.randint(0, num_classes, size=(batch_size,)), dtype=tf.int32)

    # y_pred: Prediction values (logits). Shape (14, 416)
    # The original bug specifically involved torch.float16. We test float16 here
    # to check for similar numerical or compilation issues in TensorFlow.
    y_pred = tf.constant(np.random.randn(batch_size, num_classes), dtype=tf.float16)

    # 2. Test Eager Execution
    # This corresponds to standard PyTorch eager execution.
    eager_result = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    # 3. Test Compiled Execution (tf.function)
    # This corresponds to torch.compile / torch._dynamo behavior.
    # We wrap the metric call in a tf.function to trigger graph compilation.
    @tf.function
    def compiled_metric(y_t, y_p):
        return tf.keras.metrics.sparse_categorical_accuracy(y_t, y_p)

    compiled_result = compiled_metric(y_true, y_pred)

    # 4. Verification
    # Check shapes
    assert eager_result.shape == (batch_size,), f"Eager shape mismatch: {eager_result.shape}"
    assert compiled_result.shape == (batch_size,), f"Compiled shape mismatch: {compiled_result.shape}"

    # Check for Divergence
    # The original bug was an "Eager/Compile Divergence". We assert that both modes produce the same result.
    # Note: sparse_categorical_accuracy returns 0.0 or 1.0, so float16 precision issues usually don't affect equality
    # unless there is a logic error in the graph implementation.
    are_equal = tf.reduce_all(tf.equal(eager_result, compiled_result))
    
    if not are_equal.numpy():
        print("Divergence Detected!")
        print(f"Eager Result:  {eager_result}")
        print(f"Compiled Result: {compiled_result}")
        raise AssertionError("Results differ between Eager and Compiled modes.")
    
    print("Test Passed: No divergence between Eager and Compiled modes for float16 inputs.")

if __name__ == "__main__":
    # Check for GPU availability to match the 'device=cuda' context if possible
    print(f"GPUs Available: {tf.config.list_physical_devices('GPU')}")
    test_sparse_categorical_accuracy_divergence()