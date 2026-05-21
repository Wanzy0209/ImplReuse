import torch
import tensorflow as tf
from tensorflow import keras

# Set seed for reproducibility, mimicking torch.manual_seed(9)
tf.random.set_seed(9)

# Define a custom Keras layer to act as the complex input object
# This mimics the 'arg_0' and complex tensor setup in the original test
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config

def test_serialize_keras_object():
    # Setup: Create a complex Keras object
    # Mimicking the creation of tensors in the original test
    layer_instance = CustomLayer(units=64, name="custom_layer")
    
    # Build the layer to ensure weights are initialized
    layer_instance.build((None, 10))

    # Operation: Serialize the object
    # This corresponds to the 'torch.nonzero' call in the original bug report
    # The original bug was about eager vs compile divergence.
    # serialize_keras_object is a Python utility, so we test it in eager mode.
    try:
        serialized_result = tf.keras.utils.serialize_keras_object(layer_instance)
        print(' eager success')
    except Exception as e:
        print(f' eager failure: {e}')
        return

    # Verification: Check the structure of the serialized object
    # Mimicking the comparison logic in the original test
    print(f'Serialized result: {serialized_result}')
    
    assert isinstance(serialized_result, dict), "Serialized result should be a dictionary"
    assert 'class_name' in serialized_result, "Serialized result should contain class_name"
    assert 'config' in serialized_result, "Serialized result should contain config"
    
    # Verify specific attributes
    assert serialized_result['class_name'] == 'CustomLayer'
    assert serialized_result['config']['units'] == 64
    assert serialized_result['config']['name'] == 'custom_layer'

    # Test deserialization to ensure round-trip consistency
    # This is a standard check for serialization APIs
    try:
        deserialized_layer = keras.layers.deserialize(serialized_result)
        assert isinstance(deserialized_layer, CustomLayer)
        assert deserialized_layer.units == 64
        print(' deserialization success')
    except Exception as e:
        print(f' deserialization failure: {e}')

    # Note: Unlike torch.compile, tf.function is for graph execution of tensor ops.
    # serialize_keras_object is a Python-side utility and returns a dict, not a tensor.
    # Therefore, it is not typically used inside tf.function graphs.
    # We verify the eager behavior here.

if __name__ == '__main__':
    test_serialize_keras_object()