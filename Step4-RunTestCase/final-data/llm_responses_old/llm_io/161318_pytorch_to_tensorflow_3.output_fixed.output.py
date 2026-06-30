import sys

try:
    import tensorflow as tf
    # Enable eager execution as requested by the target API.
    # This ensures operations are executed immediately, allowing for data-dependent
    # control flow and slicing without requiring a static graph compilation step.
    tf.compat.v1.enable_eager_execution()
except ImportError as e:
    print(f"Test skipped: Unable to import TensorFlow due to environment incompatibility.")
    print(f"Details: {e}")
    print("This is likely due to a missing GLIBCXX version (e.g., GLIBCXX_3.4.29) required by protobuf.")
    sys.exit(0)

def fn(encoder_attention_mask, encoder_hidden_states):
    # Replicate: encoder_hidden_states = encoder_hidden_states.new_zeros([1, 512, 3072])
    # We create a zero tensor with the same dtype as the input hidden states.
    encoder_hidden_states = tf.zeros([1, 512, 3072], dtype=encoder_hidden_states.dtype)

    # Replicate: text_len = encoder_attention_mask.sum().item()
    # In eager mode, we can compute the sum and extract the scalar value immediately.
    # tf.reduce_sum on a boolean tensor counts the number of True values.
    text_len = tf.reduce_sum(encoder_attention_mask).numpy()

    # Replicate: encoder_hidden_states = encoder_hidden_states[:, :text_len]
    # Perform the data-dependent slice using the calculated scalar length.
    encoder_hidden_states = encoder_hidden_states[:, :text_len]
    
    return encoder_hidden_states

# Setup inputs
# mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
# Create a boolean mask where the first 8 elements are True.
mask = tf.expand_dims(tf.range(512) < 8, 0)

# hidden = torch.randn((1, 512, 4096)).cuda()
hidden = tf.random.normal((1, 512, 4096))

# Run the function
result = fn(mask, hidden)

# Verify the result
# The mask sum is 8, so the slice should result in a shape of (1, 8, 3072).
expected_shape = (1, 8, 3072)
assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

print("Test passed. Data-dependent slice handled correctly in eager execution.")