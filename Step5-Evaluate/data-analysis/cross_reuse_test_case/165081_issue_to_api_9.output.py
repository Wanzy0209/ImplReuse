import tensorflow as tf
import json

# Setup similar to the PyTorch issue
# Handle TF 1.x vs 2.x API differences for setting random seed
try:
    tf.random.set_seed(52676)
except AttributeError:
    # Fallback for TensorFlow 1.x where tf.random.set_seed might not exist
    tf.set_random_seed(52676)

def test_serialize_logic():
    """
    Test case for tf.keras.layers.serialize based on the PyTorch issue 165081.
    The original issue involves an Eager/Compile divergence with complex tensor operations.
    This test verifies that tf.keras.layers.serialize produces consistent results
    between eager execution and tf.function (graph/compiled mode) when handling
    a complex layer structure.
    """
    
    # Create a complex layer structure mimicking the dimensions in the issue
    # PyTorch trace: (9, 9, 9) -> (9, 9, 11) -> (9, 11, 12) -> (9, 11, 8) -> (9, 9, 8)
    # We map this to a Keras Model structure with corresponding units.
    
    # Input layer
    inputs = tf.keras.Input(shape=(9, 9), name="input_9_9")
    
    # Layer 1: Dense to 11 units (mimicking matmul to 11)
    x = tf.keras.layers.Dense(11, name="dense_11")(inputs)
    
    # Layer 2: Dense to 12 units (mimicking matmul to 12)
    x = tf.keras.layers.Dense(12, name="dense_12")(x)
    
    # Layer 3: Dense to 8 units (mimicking matmul to 8)
    x = tf.keras.layers.Dense(8, name="dense_8")(x)
    
    # Output
    outputs = tf.keras.layers.Dense(7, name="dense_7")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Target a specific layer to serialize
    layer_to_serialize = model.layers[1] # The first Dense layer
    
    # Eager execution
    eager_result = tf.keras.layers.serialize(layer_to_serialize)
    
    # Compiled execution (tf.function)
    @tf.function
    def compiled_serialize(layer):
        return tf.keras.layers.serialize(layer)
    
    graph_result = compiled_serialize(layer_to_serialize)
    
    # Assertion: Check for divergence
    # The PyTorch issue reported a divergence where the compiler could not guard
    # on a data-dependent expression. Here we ensure the serialization is stable.
    assert eager_result == graph_result, \
        f"Eager/Compile divergence detected!\nEager: {eager_result}\nGraph: {graph_result}"
    
    return True

if __name__ == "__main__":
    result = test_serialize_logic()
    print("Test passed successfully.")