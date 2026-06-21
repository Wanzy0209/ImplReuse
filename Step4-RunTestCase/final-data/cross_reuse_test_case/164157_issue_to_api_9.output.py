import os
import tempfile
import shutil
import sys

# Handle environment dependency issues (e.g., GLIBC version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment incompatibility.")
    print(f"Error: {e}")
    sys.exit(0)

import torch

def test_fp16_std_and_save():
    """
    Test case derived from PyTorch Issue 164157.
    The original issue involves a type mismatch (fp16 vs float64) during 
    the compilation of torch.std on float16 tensors.
    
    This test translates the logic to TensorFlow, performing a reduction (std)
    on float16 tensors and saving the result using tf.data.experimental.save
    to verify type handling and compilation stability.
    """
    
    # Setup inputs similar to the PyTorch bug
    # t6, t7, t8 are float16, device=cuda (mapped to default GPU in TF if available)
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device):
        t6 = tf.random.normal([256, 88, 1], dtype=tf.float16)
        t7 = tf.random.normal([256, 88, 1], dtype=tf.float16)
        t8 = tf.random.normal([256, 88, 1], dtype=tf.float16)

        # Concatenate (mimicking t9 = torch.cat([t6, t6, t7, t8], dim=2))
        t9 = tf.concat([t6, t6, t7, t8], axis=2)

        # The operation causing the bug: std on float16
        # PyTorch: t9.std(dim=2)
        # TF: tf.math.reduce_std(t9, axis=2)
        # We wrap this in a tf.function to test compilation behavior (similar to torch.compile)
        @tf.function
        def compute_and_save(tensor_data):
            # Perform the reduction
            # Note: TF reduce_std might promote fp16 to fp32 for stability, 
            # we check if this process compiles and runs without type errors.
            t10 = tf.math.reduce_std(tensor_data, axis=2)

            # Create a dataset from the result
            dataset = tf.data.Dataset.from_tensor_slices(t10)

            # Use the similar API: tf.data.experimental.save
            # This tests if the API handles the data resulting from the fp16 operation correctly.
            path = os.path.join(tempfile.gettempdir(), "tf_fp16_std_save_test")
            
            # Clean up if exists to ensure clean run
            if os.path.exists(path):
                shutil.rmtree(path)

            tf.data.experimental.save(dataset, path)
            return t10, path

        # Execute
        try:
            result, save_path = compute_and_save(t9)
            print('Eager/Compile Success! ')
            
            # Verify output shape and basic properties
            assert result.shape == (256, 88), f"Shape mismatch: {result.shape}"
            
            # Verify the save operation created artifacts
            assert os.path.exists(save_path), "Save path does not exist"
            
            print(f"Data saved successfully to {save_path}")
            
        except Exception as e:
            print(f"Test Failed with error: {e}")
            raise

if __name__ == '__main__':
    test_fp16_std_and_save()