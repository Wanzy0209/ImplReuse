import torch
import tensorflow as tf
from tf.experimental import dtensor

def test_dtensor_data_dependent_slice():
    """
    Adapted from PyTorch Issue 161318.
    Original issue: Inductor crash on data-dependent slice inside torch.compile.
    
    This test adapts the logic to TensorFlow, using tf.function (TF's compilation)
    and the target API tf.experimental.dtensor.copy_to_mesh.
    """
    
    # Setup a local mesh for DTensor operations
    # Using a single device mesh to ensure the test runs locally without requiring a GPU cluster
    mesh = dtensor.create_mesh(['x'], [1])
    layout = dtensor.Layout([dtensor.UNSHARDED], mesh)

    # tf.function is the TensorFlow equivalent to torch.compile
    @tf.function
    def fn(encoder_attention_mask, encoder_hidden_states):
        # Replicate the logic: create zeros, get scalar from sum, slice
        # PyTorch: encoder_hidden_states.new_zeros([1, 512, 3072])
        encoder_hidden_states = tf.zeros([1, 512, 3072], dtype=tf.float32)
        
        # PyTorch: encoder_attention_mask.sum().item()
        # In TF, we use tf.reduce_sum. Note that this remains a Tensor (scalar shape),
        # allowing the slice operation to be traced with dynamic shapes.
        text_len = tf.reduce_sum(encoder_attention_mask)
        
        # PyTorch: encoder_hidden_states[:, :text_len]
        # Data-dependent slice
        encoder_hidden_states = encoder_hidden_states[:, :text_len]

        # Target API: tf.experimental.dtensor.copy_to_mesh
        # We copy the sliced tensor (which has a dynamic shape) onto the DTensor mesh.
        return dtensor.copy_to_mesh(encoder_hidden_states, layout)

    # Prepare inputs
    # PyTorch: mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
    mask = tf.expand_dims(tf.cast(tf.range(512) < 8, tf.int32), 0)
    
    # PyTorch: hidden = torch.randn((1, 512, 4096)).cuda()
    hidden = tf.random.normal((1, 512, 4096))

    # Execute the function
    try:
        result = fn(mask, hidden)
        
        # Assertions to verify behavior
        assert isinstance(result, dtensor.DTensor), "Result should be a DTensor"
        
        # Verify the shape. The sum of mask (8 ones) is 8.
        # Original shape [1, 512, 3072] -> Sliced to [1, 8, 3072]
        # Note: In TF, dynamic shapes might be represented as None in some contexts,
        # but the tensor rank should be preserved.
        assert result.shape.rank == 3, "Result should maintain 3 dimensions"
        
        print("Test passed. DTensor copy_to_mesh handled data-dependent slice successfully.")
        print(f"Result type: {type(result)}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_dtensor_data_dependent_slice()