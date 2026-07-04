import torch
import tensorflow as tf
import json

# Define a custom user-defined metric class
# This mirrors the 'Config' class in the original bug report
class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.count.assign_add(1.0)

    def result(self):
        return self.count

    def get_config(self):
        # Required for serialization to work properly
        base_config = super().get_config()
        return base_config

# Define the function to be traced/compiled
# This mirrors the 'forward' function in the original bug report
@tf.function
def serialize_metric_in_graph(metric):
    # Calling serialize() on a user-defined object inside a traced graph
    # This is the semantic equivalent of calling repr(config) inside torch.compile
    serialized = tf.keras.metrics.serialize(metric)
    return serialized

# Test execution
def test_serialize_custom_metric_in_tf_function():
    metric = CustomMetric()

    # Call the traced function
    # In the PyTorch bug, this line would raise an error regarding tracing 'repr'.
    # Here we assert it works correctly for the similar API 'serialize'.
    result = serialize_metric_in_graph(metric)

    # Assertions to verify correct behavior
    assert isinstance(result, dict), "Serialization should return a dictionary"
    assert result['class_name'] == 'CustomMetric', "Class name should be preserved"
    assert 'config' in result, "Config should be present in serialization"

    print("Test passed: tf.keras.metrics.serialize works inside tf.function.")

if __name__ == "__main__":
    test_serialize_custom_metric_in_tf_function()