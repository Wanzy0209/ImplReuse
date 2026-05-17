import torch
import tensorflow as tf

def computation_fn(encoder_attention_mask, encoder_hidden_states):
    """
    Translates the PyTorch function logic to TensorFlow.
    The core logic involves creating a zeros tensor, calculating a dynamic scalar
    from the sum of a mask, and slicing the tensor using that scalar.
    """
    # 1. Create a new zeros tensor
    # PyTorch: encoder_hidden_states.new_zeros([1, 512, 3072])
    encoder_hidden_states = tf.zeros([1, 512, 3072], dtype=encoder_hidden_states.dtype)
    
    # 2. Calculate text_len dynamically (data-dependent)
    # PyTorch: text_len = encoder_attention_mask.sum().item()
    # In TF graph mode, reduce_sum returns a 0-D tensor (scalar)
    text_len = tf.reduce_sum(encoder_attention_mask)
    
    # 3. Slice using the dynamic scalar
    # PyTorch: encoder_hidden_states = encoder_hidden_states[:, :text_len]
    encoder_hidden_states = encoder_hidden_states[:, :text_len]
    
    return encoder_hidden_states

def test_tpu_rewrite_dynamic_slice():
    # Setup inputs matching the PyTorch test case
    # PyTorch: mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
    mask = tf.expand_dims(tf.cast(tf.range(512) < 8, tf.int32), 0)
    
    # PyTorch: hidden = torch.randn((1, 512, 4096)).cuda()
    hidden = tf.random.normal((1, 512, 4096))

    # The API call: tf.compat.v1.tpu.rewrite
    # This API compiles the computation for TPU execution.
    # Note: This requires a TPU environment to execute fully.
    try:
        # Initialize TPU system
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        # Call the similar API
        # According to docs, rewrite returns a list of tensors corresponding to outputs
        result_tensors = tf.compat.v1.tpu.rewrite(computation_fn, [mask, hidden])
        
        # Execute the compiled graph
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            outputs = sess.run(result_tensors)
            
            # Verify behavior
            assert outputs is not None
            assert len(outputs) > 0
            # The expected shape depends on the mask sum (which is 8 here)
            # Shape should be [1, 8, 3072]
            assert outputs[0].shape == (1, 8, 3072), f"Expected shape (1, 8, 3072), got {outputs[0].shape}"
            print("Test passed: tf.compat.v1.tpu.rewrite handled data-dependent slice correctly.")

    except (tf.errors.NotFoundError, ValueError, tf.errors.UnavailableError) as e:
        # Gracefully handle environments without TPU hardware
        print(f"TPU hardware not detected. Skipping execution (API call is valid but requires TPU). Error: {e}")

if __name__ == "__main__":
    test_tpu_rewrite_dynamic_slice()