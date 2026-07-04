import torch
import tensorflow as tf
import numpy as np
import tempfile
import os

class ReLUInferenceWrapper(tf.keras.Model):
    """
    A wrapper model similar to HiFTInferenceWrapper in the bug report.
    It uses the similar API (tf.keras.ops.relu) in a functional manner,
    mirroring the usage of math.trunc in the original failing code.
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        # The bug report showed: call_function[target=math.trunc](args = (%mul_104,), kwargs = {})
        # We simulate a similar operation chain: a multiplication followed by the target function.
        mul_result = inputs * 2.0
        # Use the similar API: tf.keras.ops.relu
        output = tf.keras.ops.relu(mul_result)
        return output

def test_relu_export():
    """
    Test case to verify if a model using tf.keras.ops.relu can be exported
    successfully, mirroring the original bug's attempt to export a PyTorch model.
    """
    # Build example inputs mirroring the bug's tensor structure
    # Bug had: speech_feat (1, 80, mel_frames)
    batch_size = 1
    num_features = 80
    mel_frames = 10
    
    inputs = tf.random.normal((batch_size, mel_frames, num_features))

    # Instantiate the model
    model = ReLUInferenceWrapper()
    
    # Run a forward pass to build the model weights/structure
    _ = model(inputs)

    # Attempt to export the model (mirroring torch.onnx.export)
    # In TensorFlow, the standard export mechanism is saving to SavedModel format.
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "relu_model")
        try:
            model.save(export_path)
            print(f"Model successfully exported to {export_path}")
            
            # Verify the export by reloading the model
            loaded_model = tf.keras.models.load_model(export_path)
            
            # Check consistency
            original_output = model(inputs)
            loaded_output = loaded_model(inputs)
            
            assert np.allclose(original_output.numpy(), loaded_output.numpy()), \
                "Outputs differ between original and exported model"
            
            print("Test Passed: Model with tf.keras.ops.relu exported and verified successfully.")
            
        except Exception as e:
            print(f"Test Failed: {e}")
            raise

if __name__ == "__main__":
    test_relu_export()