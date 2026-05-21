import torch
import tensorflow as tf
import numpy as np

# Define a wrapper model similar to HiFTInferenceWrapper in the original issue
class ReLUInferenceWrapper(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple layer to mimic the internal processing of the HIFT module
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense(inputs)
        # Use the similar API: tf.keras.backend.relu
        # This mirrors the usage of math.trunc in the original bug report
        return tf.keras.backend.relu(x)

def build_example_inputs(batch_size=1, dim=10):
    # Build dummy inputs similar to the original build_example_inputs
    return np.random.randn(batch_size, dim).astype(np.float32)

def main():
    # 1. Setup model and inputs
    model = ReLUInferenceWrapper()
    inputs = build_example_inputs()

    # 2. Run inference to ensure the model logic is valid
    output = model(inputs)
    assert output.shape == (1, 10)

    # 3. Attempt to export the model
    # In the original issue, torch.onnx.export failed with math.trunc.
    # Here we test if tf.keras.backend.relu allows successful export to TFLite.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    try:
        tflite_model = converter.convert()
        # Assert that the export produced a valid model buffer
        assert tflite_model is not None
        print("Test passed: Model with tf.keras.backend.relu exported successfully.")
    except Exception as e:
        print(f"Test failed: Export error encountered - {e}")
        raise

if __name__ == "__main__":
    main()