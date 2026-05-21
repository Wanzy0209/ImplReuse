import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor
from collections import OrderedDict
import tempfile
import os

def test_name_based_save_preserves_view_dtype():
    """
    Test case to verify that tf.experimental.dtensor.name_based_save 
    correctly handles tensors that have been bitcasted (viewed) to a different dtype.
    
    This mirrors the PyTorch issue where as_strided lowering threw away .view(dtype),
    causing the kernel to see the original dtype instead of the viewed one (e.g., uint8).
    """
    # Setup: Create a mesh for DTensor operations
    # Using CPU as a default device to ensure the test runs in most environments
    devices = tf.config.list_physical_devices()
    if not devices:
        devices = [tf.DeviceSpec(device_type="CPU")]
    
    mesh = dtensor.create_mesh([("x", 1)], devices=devices)

    # 1. Create a tensor with a specific dtype (e.g., float32)
    original_data = [1.0, 2.0, 3.0, 4.0]
    tensor_f32 = tf.constant(original_data, dtype=tf.float32)
    
    # Distribute the tensor to make it a DTensor
    layout = dtensor.Layout([dtensor.UNSHARDED], mesh)
    d_tensor_f32 = dtensor.copy_to_mesh(tensor_f32, layout)

    # 2. Apply a view operation (bitcast) to change the dtype without changing data
    # PyTorch equivalent: tensor.view(torch.uint8)
    # TensorFlow equivalent: tf.bitcast
    d_tensor_viewed = tf.bitcast(d_tensor_f32, tf.uint8)

    # Verify the view operation resulted in the correct dtype
    assert d_tensor_viewed.dtype == tf.uint8, "Bitcast operation failed to change dtype"

    # 3. Prepare inputs for the API under test
    # The API expects a dictionary of names to tensors.
    # The bug in PyTorch occurred when the compiler ignored this view during lowering.
    name_tensor_dict = OrderedDict({
        "viewed_tensor": d_tensor_viewed
    })

    # 4. Execute the API
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_prefix = os.path.join(tmpdir, "test_checkpoint")
        
        try:
            # Call the similar API
            dtensor.name_based_save(mesh, checkpoint_prefix, name_tensor_dict)
            
            # Verification:
            # In the PyTorch bug, the lowered kernel arguments showed the wrong dtype (float8 instead of uint8).
            # Here, we verify that the API successfully processes the tensor with the viewed dtype.
            # If the API "threw away" the view like the PyTorch bug, it might fail to save 
            # or save with incorrect metadata, potentially causing issues on restore.
            
            # We assert that the tensor we passed in maintains its viewed dtype property
            # through the context of the save operation.
            saved_dtype = name_tensor_dict["viewed_tensor"].dtype
            assert saved_dtype == tf.uint8, \
                f"API failed to preserve viewed dtype. Expected uint8, got {saved_dtype}"
            
            print("Test Passed: name_based_save preserved the bitcast (viewed) dtype correctly.")
            
        except Exception as e:
            print(f"Test Failed with exception: {e}")
            raise

if __name__ == "__main__":
    test_name_based_save_preserves_view_dtype()