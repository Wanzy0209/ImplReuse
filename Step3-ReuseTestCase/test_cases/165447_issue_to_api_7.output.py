import tensorflow as tf
import torch

def test_to_dlpack_multiple_runs():
    """
    Test case for tf.experimental.dlpack.to_dlpack inspired by Issue 165447.
    
    The original issue involves serializing a compiled model, saving it, 
    resetting the environment, loading it back, and verifying correctness, 
    specifically checking for failures when running this process multiple times.
    
    This test adapts that logic to the DLPack API:
    1. Create a TensorFlow tensor (analogous to the model/input).
    2. Serialize it to a DLPack capsule using to_dlpack (analogous to aot_compile/save).
    3. Consume the capsule (load it into PyTorch) to verify data transfer.
    4. Repeat the process to ensure the serialization mechanism handles multiple runs correctly.
    """
    
    # Create a sample tensor (analogous to sample_inputs in the bug report)
    original_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

    # Run the serialization/deserialization flow multiple times
    # to address the "running multiple times" aspect of the bug title.
    for i in range(2):
        # Serialize the tensor to a DLPack capsule.
        # This is analogous to compiled_fn.save_compiled_function(compiled_fn_path).
        dlpack_capsule = tf.experimental.dlpack.to_dlpack(original_tensor)

        # Load/Consume the capsule.
        # We use PyTorch to consume the capsule, which is the standard use case for DLPack.
        # This is analogous to torch.compiler.load_compiled_function(f).
        loaded_tensor = torch.from_dlpack(dlpack_capsule)

        # Verify the data integrity.
        # This is analogous to assert torch.allclose(eager_out, compiled_out).
        # We convert the PyTorch tensor back to a TensorFlow tensor for easy comparison.
        recovered_tensor = tf.convert_to_tensor(loaded_tensor.numpy())
        
        assert tf.reduce_all(tf.equal(original_tensor, recovered_tensor)).numpy(), \
            f"Data mismatch on run {i+1}"

if __name__ == "__main__":
    test_to_dlpack_multiple_runs()
    print("Test passed: DLPack serialization successful over multiple runs.")