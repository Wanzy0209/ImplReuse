```python
import tensorflow as tf

# PyTorch: torch.device("mps")
# TensorFlow uses device strings. On Apple Silicon with Metal support, 
# the primary GPU is typically accessed via "/GPU:0".
device = "/GPU:0"

with tf.device(device):
    # PyTorch: torch.zeros(10, dtype=torch.complex64, device=device)
    # Use tf.Variable to allow in-place modification (index_put_)
    image = tf.Variable(tf.zeros(10, dtype=tf.complex64))
    
    # PyTorch: torch.ones(3, dtype=torch.complex64, device=device)
    data = tf.ones(3, dtype=tf.complex64)
    
    # PyTorch: torch.tensor([1, 3, 5], device=device)
    # TensorFlow scatter operations require indices to be shaped (num_updates, rank).
    # For a 1D tensor, we reshape the indices to (3, 1).
    indices = tf.constant([[1], [3], [5]], dtype=tf.int32)

    # PyTorch: image.index_put_(indices=(indices,), values=data, accumulate=True)
    # TensorFlow: tf.Variable.scatter_nd_add performs the in-place accumulation.
    # Note: indices must be expanded to match the rank of the tensor being updated.
    image.scatter_nd_add(indices, data)
```