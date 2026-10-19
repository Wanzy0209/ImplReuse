```python
import tensorflow as tf

def test_index_select(device):
    # Map PyTorch device strings to TensorFlow device strings
    # "mps" in PyTorch typically maps to GPU in TensorFlow on Apple Silicon
    tf_device = '/CPU:0' if device == 'cpu' else '/GPU:0'

    with tf.device(tf_device):
        # torch.ones([2, 3], device=device) -> tf.ones([2, 3])
        x = tf.ones([2, 3])

        # torch.tensor(1, device=device) -> tf.constant(1)
        # Note: PyTorch index_select typically requires a 1D index tensor, 
        # but the source code provides a 0-d tensor. 
        # TensorFlow's gather handles scalar indices correctly.
        index = tf.constant(1)

        try:
            # torch.index_select(x, dim=0, index=index) -> tf.gather(x, index, axis=0)
            output = tf.gather(x, index, axis=0)
            print(f"index_select test succeeds for device: {device}. output shape: {output.shape}")
        except Exception as e:
            print(f"index_select test fails for device: {device}: {e}")

test_index_select(device = "cpu")
test_index_select(device = "mps")
```