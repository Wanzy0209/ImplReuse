import torch
import tensorflow as tf
from collections import namedtuple
import tempfile
import shutil

def test_snapshot_namedtuple():
    """
    Adapted test case for tf.data.experimental.snapshot based on the 
    PyTorch export bug regarding NamedTuple inputs.
    
    The original bug highlights that torch.export.export fails with 
    NamedTuple inputs in non-strict mode. This test verifies that 
    tf.data.experimental.snapshot correctly handles datasets yielding 
    NamedTuple structures.
    """
    # Define the NamedTuple structure (similar to PyTorch example)
    Point = namedtuple('Point', ['x', 'y'])

    # Create a dataset that yields NamedTuples
    # This simulates the 'inp = Point(...)' input scenario
    def gen_data():
        for i in range(3):
            # Using tf.ones to match the original torch.ones(3) logic
            yield Point(tf.ones(3) * i, tf.ones(3) * (i + 1))

    # Define output signature matching the NamedTuple fields
    output_signature = (
        tf.TensorSpec(shape=(3,), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(gen_data, output_signature=output_signature)

    # Setup temporary directory for snapshot
    snapshot_dir = tempfile.mkdtemp()

    try:
        # Call the API: tf.data.experimental.snapshot
        # This corresponds to 'ep = torch.export.export(...)' in the original bug.
        # We expect this to handle the NamedTuple structure without error.
        snapshotted_dataset = dataset.snapshot(snapshot_dir)

        # Verify the result by consuming the dataset
        # This corresponds to checking the validity of the exported program
        results = list(snapshotted_dataset.as_numpy_iterator())
        
        # Assertions to ensure correctness
        assert len(results) == 3, "Expected 3 items in the dataset"
        
        # Verify the content of the first item
        # Note: TF might deserialize back to a tuple/list rather than the specific 
        # NamedTuple class, but the structure and values must be preserved.
        item_0 = results[0]
        assert len(item_0) == 2, "Expected 2 fields in the output structure"
        
        # Check values (x=0, y=1 for the first iteration)
        assert (item_0[0] == 0).all(), "Field 'x' mismatch"
        assert (item_0[1] == 1).all(), "Field 'y' mismatch"

        print("Test passed: tf.data.experimental.snapshot handled NamedTuple inputs correctly.")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(snapshot_dir)

if __name__ == "__main__":
    test_snapshot_namedtuple()