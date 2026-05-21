import torch
import tensorflow as tf
import os
import shutil
import tempfile

# Define the first dataset pipeline (mimicking AnyDimsModelEmpty)
# This corresponds to torch.ops.aten.any.dims(x, [], False)
# In TensorFlow, axis=[] implies reducing over no dimensions (identity operation regarding shape)
def get_dataset_empty_axis():
    data = tf.constant([[True, False], [False, False]])
    ds = tf.data.Dataset.from_tensors(data)
    return ds.map(lambda x: tf.reduce_any(x, axis=[]))

# Define the second dataset pipeline (mimicking AnyDimsModelNull)
# This corresponds to torch.ops.aten.any.dims(x, None, False)
# In TensorFlow, axis=None implies reducing over all dimensions (scalar result)
def get_dataset_null_axis():
    data = tf.constant([[True, False], [False, False]])
    ds = tf.data.Dataset.from_tensors(data)
    return ds.map(lambda x: tf.reduce_any(x, axis=None))

def process_dataset(ds, path, name):
    print(f"Processing: {name}")
    
    # 1. Run eager mode
    print("Running eager mode...")
    eager_shape = None
    for elem in ds.take(1):
        eager_shape = elem.shape
        print(f"Eager output shape: {eager_shape}")

    # 2. Snapshot (Export)
    print(f"Snapshotting to {path}...")
    # Clean path to ensure fresh state
    if os.path.exists(path):
        shutil.rmtree(path)
        
    snapshot_ds = ds.snapshot(path)
    
    # 3. Verify Snapshot shape
    print("Verifying snapshot output...")
    snapshot_shape = None
    for elem in snapshot_ds.take(1):
        snapshot_shape = elem.shape
        print(f"Snapshot output shape: {snapshot_shape}")

    # Check for the bug: The snapshot shape should match the eager shape.
    # In the PyTorch bug, the second export would incorrectly inherit the shape of the first.
    assert eager_shape == snapshot_shape, \
        f"BUG DETECTED: Shape mismatch for {name}. Eager: {eager_shape}, Snapshot: {snapshot_shape}"
    print()

if __name__ == "__main__":
    # Setup temporary directories for snapshots
    base_dir = tempfile.mkdtemp()
    path_empty = os.path.join(base_dir, "snapshot_empty")
    path_null = os.path.join(base_dir, "snapshot_null")

    try:
        # Process the first model (Empty Axis)
        # Expected shape: (2, 2)
        ds_empty = get_dataset_empty_axis()
        process_dataset(ds_empty, path_empty, "EmptyAxisModel")

        # Process the second model (Null Axis)
        # Expected shape: () (Scalar)
        # In the original PyTorch bug, running this after the first model 
        # would result in an incorrect shape (e.g., (2, 2) instead of ()).
        ds_null = get_dataset_null_axis()
        process_dataset(ds_null, path_null, "NullAxisModel")
        
        print("Test passed. No shape mismatches detected.")
        
    finally:
        # Cleanup
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)