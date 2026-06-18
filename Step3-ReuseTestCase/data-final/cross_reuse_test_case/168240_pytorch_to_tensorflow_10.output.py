import torch
import tensorflow as tf
import numpy as np
import tempfile
import os

def test_spatial_dropout1d_export():
    """
    Adapted test case for tf.keras.layers.SpatialDropout1D.
    Verifies that a model containing this layer maintains consistent behavior
    after being exported (saved) and reloaded, mirroring the logic of the
    original torch.export.export bug report.
    """
    # 1. Define a model using the target API: tf.keras.layers.SpatialDropout1D
    # SpatialDropout1D expects a 3D input: (batch_size, timesteps, channels)
    # We create a simple sequential model to encapsulate the layer.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(10, 32)),
        tf.keras.layers.SpatialDropout1D(rate=0.2),
        tf.keras.layers.Dense(10)
    ])

    # 2. Create dummy input
    # Shape matches the InputLayer defined above (batch=1, timesteps=10, channels=32)
    x = np.random.rand(1, 10, 32).astype(np.float32)

    # 3. Get output from the original model
    # We use training=False to ensure deterministic output for the comparison,
    # similar to how one would typically verify an exported model's correctness in inference mode.
    original_output = model(x, training=False)

    # 4. "Export" the model
    # In TensorFlow/Keras, the equivalent of exporting a program is saving the model.
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'saved_model')
        model.save(model_path)

        # 5. Load the exported model
        loaded_model = tf.keras.models.load_model(model_path)

        # 6. Get output from the loaded model
        loaded_output = loaded_model(x, training=False)

        # 7. Verify behavior
        # Assert that the outputs are close, mirroring torch.testing.assert_close
        try:
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)
            print("Test passed: Exported model behavior matches original.")
        except AssertionError as e:
            print(f"Test failed: {e}")
            raise

if __name__ == "__main__":
    test_spatial_dropout1d_export()