import sys
import tempfile
import shutil
import numpy as np

# Handle environment incompatibility for TensorFlow (e.g., GLIBC version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    if "GLIBC" in str(e) or "libstdc++" in str(e):
        print("Test skipped: TensorFlow environment is incompatible (GLIBC/libstdc++ version mismatch).")
        print(f"Error details: {e}")
        sys.exit(0)
    else:
        raise

def test_tf_data_snapshot_integrity():
    """
    Adapts the logic of the torch.export.export bug report to tf.data.experimental.snapshot.
    
    Original Logic:
    1. Create a model (MobileNetV2).
    2. Create input data.
    3. Export the model (torch.export.export).
    4. Compare the output of the original model vs the exported model.
    
    Adapted Logic:
    1. Create a data pipeline (Dataset with a map function acting as the 'model').
    2. Create input data.
    3. Snapshot the pipeline (tf.data.experimental.snapshot).
    4. Compare the output of the original pipeline vs the snapshot pipeline.
    """
    
    # Setup a temporary directory for the snapshot
    snapshot_dir = tempfile.mkdtemp()

    try:
        # 1. Create input data (similar to torch.rand((1, 3, 224, 224)))
        # We use a small batch size for the test
        input_data = tf.random.uniform((10, 3, 224, 224))
        dataset = tf.data.Dataset.from_tensor_slices(input_data)

        # 2. Define a transformation function (similar to the model forward pass)
        # We use a deterministic arithmetic operation to represent the model logic
        def model_like_transform(x):
            return x * 2.0 - 1.0

        # Apply the transformation to get the "ground truth" pipeline
        original_dataset = dataset.map(model_like_transform)

        # 3. Apply the API: tf.data.experimental.snapshot
        # This persists the output of the pipeline to disk, analogous to exporting a static graph.
        snapshot_dataset = original_dataset.snapshot(snapshot_dir)

        # 4. Verify behavior (similar to torch.testing.assert_close)
        # We need to iterate through the datasets to materialize the data
        # and ensure the snapshot preserves the exact state of the pipeline.
        original_results = list(original_dataset.as_numpy_iterator())
        snapshot_results = list(snapshot_dataset.as_numpy_iterator())

        # Assert that the snapshot preserves the data correctly
        # This mirrors the logic: assert model(x) == ep.module()(x)
        for orig, snap in zip(original_results, snapshot_results):
            np.testing.assert_allclose(orig, snap, rtol=1e-5, atol=1e-5, 
                                       err_msg="Snapshot data mismatch: The snapshot pipeline produced different results than the original pipeline.")

        print("Test passed: Snapshot data matches original data.")

    finally:
        # Cleanup the temporary directory
        shutil.rmtree(snapshot_dir)

if __name__ == "__main__":
    test_tf_data_snapshot_integrity()