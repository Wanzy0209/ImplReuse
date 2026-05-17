import tensorflow as tf

# Enable eager execution to mimic PyTorch's immediate execution mode
tf.compat.v1.enable_eager_execution()

def test_dropout(device="/cpu:0"):
    with tf.device(device):
        x = tf.ones([2, 3])
        # In the original bug, 'index' is a scalar tensor (0-dim).
        # Here, 'noise_shape' is the argument that defines structure.
        # We test if dropout handles a 0-dim tensor for noise_shape.
        noise_shape = tf.constant(1)  # zero-dimensional tensor

        try:
            output = tf.compat.v1.nn.dropout(x, rate=0.5, noise_shape=noise_shape)
            print(f"dropout test succeeds for device: {device}. output shape: {output.shape}")
        except Exception as e:
            print(f"dropout test fails for device: {device}: {e}")

# Test on CPU
test_dropout(device="/cpu:0")

# Test on GPU if available (analogous to MPS)
if tf.config.list_physical_devices('GPU'):
    test_dropout(device="/gpu:0")
else:
    print("GPU not available, skipping GPU test.")