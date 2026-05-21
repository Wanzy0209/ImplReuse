import torch
import tensorflow as tf
import numpy as np

# Define a custom loss that uses atan2, mirroring the bug report's context
# where a model (or loss function) uses atan2.
class Atan2Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # The bug report highlights the behavior of atan2(0, 0)
        return tf.math.atan2(y_true, y_pred)

    def get_config(self):
        return super().get_config()

def test_atan2_serialization_consistency():
    """
    Test that tf.keras.losses.serialize preserves the behavior of atan2
    for edge case inputs (0, 0), similar to the PyTorch/ONNX issue.
    """
    # Inputs corresponding to the bug report (0, 0)
    x = tf.constant([0.0])
    y = tf.constant([0.0])

    # Native execution
    loss = Atan2Loss()
    native_result = loss(x, y)

    # Serialization using the similar API (tf.keras.losses.serialize)
    # This corresponds to the 'export' step in the original bug report.
    config = tf.keras.losses.serialize(loss)

    # Deserialization corresponds to loading the model in ONNX Runtime
    deserialized_loss = tf.keras.losses.deserialize(config)
    deserialized_result = deserialized_loss(x, y)

    print("Native result:", native_result.numpy())
    print("Deserialized result:", deserialized_result.numpy())

    # In the original bug, torch.atan2(0, 0) returned 0, but the ONNX export returned NaN.
    # We verify that TF serialization preserves the native behavior (0).
    assert not np.isnan(native_result.numpy()[0]), "Native result should not be NaN"
    assert not np.isnan(deserialized_result.numpy()[0]), "Deserialized result should not be NaN"
    assert np.allclose(native_result.numpy(), deserialized_result.numpy()), \
        "Serialization should preserve atan2 behavior for (0, 0)"

if __name__ == "__main__":
    test_atan2_serialization_consistency()