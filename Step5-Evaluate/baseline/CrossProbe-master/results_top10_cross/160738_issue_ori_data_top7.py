```python
import tensorflow as tf

def test_var(device):
    # Map PyTorch device strings to TensorFlow device strings
    # PyTorch "cpu" -> TF "/cpu:0"
    # PyTorch "mps" (Metal) -> TF "/device:GPU:0" (assuming GPU/Metal support)
    tf_device = "/cpu:0" if device == "cpu" else "/device:GPU:0"

    with tf.device(tf_device):
        # Conversion: torch.tensor(3.0, device=device) -> tf.constant(3.0)
        x = tf.constant(3.0)
        try:
            # Conversion: torch.var(x, dim=0) -> tf.math.reduce_variance(x, axis=0)
            # Note: tf.math.reduce_variance computes population variance by default, matching torch.var
            output = tf.math.reduce_variance(x, axis=0)
            print(f"var test succeeds for device: {device}. output: {output}")
        except Exception as e:
            print(f"var test fails for device: {device}: {e}")

test_var(device = "cpu")
test_var(device = "mps")
```