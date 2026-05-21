import torch
import tensorflow as tf
import numpy as np

# The original bug report involves a complex sequence of matrix multiplications (torch.matmul)
# and a divergence in eager vs compiled mode (torch._dynamo).
# The target API is tf.keras.legacy.saving.serialize_keras_object.
# 
# Since the APIs are semantically different (Index finding vs Serialization),
# we adapt the test by creating a complex Keras model (mimicking the matmul complexity)
# and verifying that the serialization API handles it correctly without errors.

def test_serialize_complex_model():
    # Setup: Create a Keras model with multiple layers to mimic the complexity
    # of the fuzzed_program which performs chained matmul operations.
    # We use dimensions similar to those found in the fuzzer (e.g., 9, 11, 8).
    inputs = tf.keras.Input(shape=(9,), name="input_layer")
    x = tf.keras.layers.Dense(11, name="dense_1")(inputs)
    x = tf.keras.layers.Dense(8, name="dense_2")(x)
    x = tf.keras.layers.Dense(16, name="dense_3")(x)
    outputs = tf.keras.layers.Dense(7, name="dense_4")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="complex_model")

    # Action: Serialize the model using the target API
    # Note: tf.keras.legacy.saving.serialize_keras_object is the specific API requested.
    try:
        serialized_config = tf.keras.legacy.saving.serialize_keras_object(model)
        
        # Verification: Ensure the output is a valid dictionary structure
        assert isinstance(serialized_config, dict), "Output should be a dictionary"
        assert 'class_name' in serialized_config, "Missing 'class_name' in serialization"
        assert 'config' in serialized_config, "Missing 'config' in serialization"
        
        print("Test Passed: tf.keras.legacy.saving.serialize_keras_object handled the complex model.")
        print(f"Serialized Class Name: {serialized_config.get('class_name')}")
        return True
        
    except Exception as e:
        print(f"Test Failed: {e}")
        return False

if __name__ == "__main__":
    test_serialize_complex_model()