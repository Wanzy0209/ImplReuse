import tensorflow as tf
from collections import namedtuple

def test_namedtuple():
    # Enable eager execution to ensure .numpy() works on tensors.
    # This is necessary for environments where TF might be running in graph mode (TF1 compatibility mode).
    tf.compat.v1.enable_eager_execution()

    # Define a NamedTuple for input data (similar to the original PyTorch test)
    Point = namedtuple('Point', 'x y')
    
    # Define a NamedTuple for input shapes (metadata for tf.keras.Input)
    InputShape = namedtuple('InputShape', 'x y')

    # Create input data
    inp_data = Point(tf.ones(3), tf.ones(3))
    
    # Create input shape specification
    inp_shape = InputShape((3,), (3,))

    # Test Case 1: Attempting to pass the NamedTuple directly to tf.keras.Input
    # This mirrors the "errors" case in the original bug report where direct usage failed.
    # tf.keras.Input expects a shape tuple (integers), not a NamedTuple object.
    try:
        # This should fail because tf.keras.Input does not accept a NamedTuple as the shape argument
        x = tf.keras.Input(shape=inp_data, name='x')
        assert False, "tf.keras.Input should not accept a NamedTuple directly as shape"
    except (TypeError, ValueError) as e:
        print(f"Expected error with direct NamedTuple input: {e}")

    # Test Case 2: Using the fields of the NamedTuple correctly
    # This mirrors the "succeeds" case.
    # We unpack the NamedTuple to provide the necessary arguments to tf.keras.Input.
    x = tf.keras.Input(shape=inp_shape.x, name='x')
    y = tf.keras.Input(shape=inp_shape.y, name='y')
    
    # Define a simple model (similar to class M in the original)
    # In Keras Functional API, we define the graph structure
    output = x + y
    
    model = tf.keras.Model(inputs=[x, y], outputs=output)
    
    # Test calling the model with the NamedTuple data
    # This verifies that the inputs defined via tf.keras.Input work with the NamedTuple structure
    result = model(inp_data)
    
    # Verify the result
    # The original test printed the result, we will assert it matches the expected output (2.0)
    assert result.shape == (3,)
    assert tf.reduce_all(result == 2.0).numpy()
    
    print("Test passed: tf.keras.Input works correctly with NamedTuple fields.")

if __name__ == "__main__":
    test_namedtuple()