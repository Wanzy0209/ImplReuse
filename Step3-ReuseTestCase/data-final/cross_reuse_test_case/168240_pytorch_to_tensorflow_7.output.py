import torch
import tensorflow as tf
import numpy as np
import tempfile
import os

def test_tf_keras_input_export_consistency():
    """
    Adapted test case for tf.keras.Input based on the torch.export.export bug.
    Verifies that a model defined with tf.keras.Input maintains consistency
    after being exported (saved) and loaded.
    """
    # 1. Define the input using the target API: tf.keras.Input
    # This matches the shape used in the original PyTorch bug report (1, 3, 224, 224)
    # Note: TensorFlow uses (Height, Width, Channels) for shape, excluding batch size in Input definition usually,
    # but we can specify batch_size if needed. Here we stick to standard functional API usage.
    inputs = tf.keras.Input(shape=(224, 224, 3), batch_size=1)

    # 2. Create the model (MobileNetV2) using the defined input
    # We use input_tensor to connect our explicit tf.keras.Input layer to the application model.
    # weights=None ensures we are testing the architecture/graph, not pre-trained accuracy.
    model = tf.keras.applications.MobileNetV2(weights=None, input_tensor=inputs)

    # 3. Generate random input data matching the original bug report
    x = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # 4. Run the original model
    y_original = model(x)

    # 5. "Export" the model
    # In TensorFlow/Keras, the equivalent of exporting a program is saving the model (e.g., to SavedModel format).
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "mobilenet_v2_exported")
        model.save(model_path)

        # 6. Load the exported model
        loaded_model = tf.keras.models.load_model(model_path)

        # 7. Run the exported model
        y_exported = loaded_model(x)

    # 8. Verify consistency
    # The original bug showed a mismatch between the original and exported model.
    # We assert that the outputs are close to ensure the export process preserved the graph logic.
    np.testing.assert_allclose(y_original, y_exported, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_tf_keras_input_export_consistency()
    print("Test passed: Exported model matches original model.")