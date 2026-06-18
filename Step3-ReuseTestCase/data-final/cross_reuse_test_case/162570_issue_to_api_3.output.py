import torch
import tensorflow as tf
import numpy as np

# This test case mirrors the logic of the original bug report:
# Original: torch.atan2(0, 0) returns 0, but ONNX export returns NaN.
# Adaptation: tf.keras.initializers.Constant(0.0) should return 0,
# and its serialized/deserialized form should also return 0.

# Define the component (analogous to Net)
# Using Constant(0.0) to mirror the zero inputs in the atan2 bug.
initializer = tf.keras.initializers.Constant(value=0.0)

# Native execution
tf_result = initializer(shape=(1,))

# Export (Serialize)
config = tf.keras.initializers.serialize(initializer)

# Import (Deserialize) - analogous to loading the ONNX session
restored_initializer = tf.keras.initializers.deserialize(config)

# Exported execution
restored_result = restored_initializer(shape=(1,))

print("tf_result", tf_result.numpy()) # Expected: [0.]
print("restored_result", restored_result.numpy()) # Expected: [0.]

# Assert that the serialization preserves the behavior exactly.
# In the original atan2 bug, this comparison would fail (0.0 vs NaN).
np.testing.assert_array_equal(tf_result.numpy(), restored_result.numpy())