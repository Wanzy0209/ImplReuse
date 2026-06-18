import tensorflow as tf
from google.protobuf import json_format

# Wrapper over the tf.io.decode_json_example operation.
# This mirrors the MPSSoftshrink class in the original bug report,
# wrapping a low-level operation for use in a high-level model.
class JSONDecodeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(JSONDecodeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs are JSON-encoded strings
        return tf.io.decode_json_example(inputs)

# Wrapper over the Sequential layer, using the custom JSON decode implementation.
# This mirrors the CustomMPSSoftshrinkModel class structure.
class CustomJSONDecodeModel(tf.keras.Model):
    def __init__(self):
        super(CustomJSONDecodeModel, self).__init__()
        self.model = tf.keras.Sequential([
            JSONDecodeLayer(),
            # Add a parsing step to make the model functional, similar to Linear layers
            # in the original PyTorch code.
            tf.keras.layers.Lambda(lambda x: tf.io.parse_example(
                x,
                features={
                    "a": tf.io.FixedLenFeature([], tf.int64),
                    "b": tf.io.FixedLenFeature([], tf.float32)
                }
            ))
        ])

    def call(self, x):
        return self.model(x)

# Test execution
if __name__ == "__main__":
    # Create dummy data
    example_proto = tf.train.Example(features=tf.train.Features(feature={
        "a": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        "b": tf.train.Feature(float_list=tf.train.FloatList(value=[2.0]))
    }))
    
    # Convert to JSON string as required by the API
    json_string = json_format.MessageToJson(example_proto)
    
    # Create a batch of inputs
    inputs = tf.constant([json_string, json_string])

    # Instantiate and run the model
    model = CustomJSONDecodeModel()
    output = model(inputs)

    # Assertions to verify correct behavior and stability
    assert output is not None
    assert "a" in output
    assert "b" in output
    assert output["a"].shape == (2,)
    assert output["b"].shape == (2,)
    
    print("Test passed: tf.io.decode_json_example integration successful.")