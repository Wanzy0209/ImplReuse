import torch
import tensorflow as tf
import tempfile
import shutil
import os

def test_tf_snapshot_dynamic_slicing():
    """
    Adapts the PyTorch dynamic slicing bug (Issue 163146) to TensorFlow's 
    tf.data.experimental.snapshot.

    Original Bug Context:
    torch.export.export failed with a "Data dependent error" when encountering
    a slice operation where the stop index was a Tensor:
    `selected_item_embedding = item_embedding[:, :max_item_num, :]`

    This test verifies if tf.data.experimental.snapshot can correctly persist
    a dataset pipeline that performs similar dynamic slicing operations.
    """
    
    # Create a temporary directory for the snapshot
    snapshot_path = tempfile.mkdtemp()

    try:
        # 1. Prepare Data
        # Mimicking the inputs from the PyTorch traceback:
        # item_embedding: Tensor(shape: [s10, s64, 64])
        # max_item_num: Tensor(shape: [])
        
        # We create a dataset that yields batches of embeddings and a dynamic limit (scalar tensor)
        def data_generator():
            for _ in range(2):
                # Shape: [10, 64, 64]
                embedding = tf.random.uniform((10, 64, 64))
                # Shape: [] (Scalar), dynamic value between 10 and 64
                # This represents the 'max_item_num' which caused the data dependency issue
                limit = tf.random.uniform([], minval=10, maxval=64, dtype=tf.int32)
                yield embedding, limit

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(10, 64, 64), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

        # 2. Define the Dynamic Slicing Logic
        # This corresponds to the line in the PyTorch traceback:
        # selected_item_embedding = item_embedding[:, :max_item_num, :]
        def apply_dynamic_slicing(embedding, max_item_num):
            # Perform slicing where the stop index is a Tensor.
            # This is the core logic that triggered the "Data dependent error" in PyTorch.
            return embedding[:, :max_item_num, :]

        # 3. Construct the Pipeline
        # Apply the slicing operation
        sliced_dataset = dataset.map(apply_dynamic_slicing)
        
        # Apply the snapshot API (The Similar API)
        # This attempts to persist the output of the dynamic slicing operation to disk.
        snapshot_dataset = sliced_dataset.snapshot(snapshot_path)

        # 4. Execute / Verify
        # In PyTorch, the error occurred during the export attempt.
        # In TF, the snapshot is materialized when we iterate the dataset.
        print("Attempting to materialize snapshot with dynamic slicing...")
        results = list(snapshot_dataset.as_numpy_iterator())
        
        # Assertions to verify correctness
        assert len(results) == 2, "Expected 2 batches in the snapshot"
        
        for i, res in enumerate(results):
            # The second dimension should be <= 64 because of the dynamic slice
            # res shape is [10, dynamic_limit, 64]
            assert res.shape[0] == 10, f"Batch dim mismatch in batch {i}"
            assert res.shape[2] == 64, f"Feature dim mismatch in batch {i}"
            assert res.shape[1] <= 64, f"Sliced dim too large in batch {i}"
            print(f"Batch {i} shape: {res.shape} - Snapshot successful.")

        print("Test Passed: tf.data.experimental.snapshot handled dynamic slicing.")

    finally:
        # Cleanup temporary directory
        if os.path.exists(snapshot_path):
            shutil.rmtree(snapshot_path)

if __name__ == "__main__":
    test_tf_snapshot_dynamic_slicing()