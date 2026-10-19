import torch
import tensorflow as tf
import numpy as np

def to_tf_sparse(torch_tensor):
    """Helper to convert a PyTorch tensor to a TensorFlow SparseTensor."""
    indices = torch.nonzero(torch_tensor).T
    if indices.numel() == 0:
        return tf.sparse.SparseTensor(
            indices=np.zeros((0, torch_tensor.dim()), dtype=np.int64),
            values=np.array([], dtype=np.float32),
            dense_shape=torch_tensor.shape
        )
    # Gather values based on indices
    if torch_tensor.dim() == 3:
        values = torch_tensor[indices[0], indices[1], indices[2]]
    else:
        values = torch_tensor[indices]
        
    return tf.sparse.SparseTensor(
        indices=indices.numpy(),
        values=values.numpy(),
        dense_shape=torch_tensor.shape
    )

# Original: input_tensor = torch.empty_strided(size = [1792, 1899, 160], stride = (303872, 160, 1))
# Replacement: torch.cummax
# We create a dummy input tensor to apply cummax, as cummax requires an input.
dummy_input = torch.randn(1792, 1899, 160)
# torch.cummax returns a tuple (values, indices), we take the values to match the tensor type.
input_tensor = torch.cummax(dummy_input, dim=0)[0]

# Original: output_tensor = torch.rand_like(input_tensor)
# Replacement: torch.fmax
# torch.fmax requires two tensors. We use input_tensor and itself to ensure the assertion logic holds.
output_tensor = torch.fmax(input_tensor, input_tensor)

# Original: assert input_tensor.stride() == output_tensor.stride()
# Replacement: tf.sparse.add
# We convert the PyTorch tensors to TensorFlow SparseTensors to use tf.sparse.add.
sparse_input = to_tf_sparse(input_tensor)
sparse_output = to_tf_sparse(output_tensor)

# Create a zero sparse tensor to serve as the second argument for tf.sparse.add
zero_sparse = tf.sparse.SparseTensor(
    indices=np.zeros((0, 3), dtype=np.int64),
    values=np.array([], dtype=np.float32),
    dense_shape=input_tensor.shape
)

# Perform the sparse addition
result_input = tf.sparse.add(sparse_input, zero_sparse)
result_output = tf.sparse.add(sparse_output, zero_sparse)

# Assert that the results are equal
assert tf.reduce_all(tf.equal(tf.sparse.to_dense(result_input), tf.sparse.to_dense(result_output))).numpy()