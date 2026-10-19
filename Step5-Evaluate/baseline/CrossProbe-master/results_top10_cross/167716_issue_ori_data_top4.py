```python
import tensorflow as tf

# Create dense tensor 'a' and convert to sparse
# PyTorch: torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse()
a_dense = tf.constant([[1., 0, 2], [0, 3, 0]], dtype=tf.float32)
a_indices = tf.where(tf.not_equal(a_dense, 0))
a_values = tf.gather_nd(a_dense, a_indices)
# TensorFlow SparseTensors require indices to be sorted in row-major order
a = tf.sparse.reorder(tf.sparse.SparseTensor(a_indices, a_values, dense_shape=a_dense.shape))
# Note: PyTorch's .requires_grad_() is not directly supported on TF SparseTensors

# Create dense tensor 'b' with gradient tracking
# PyTorch: torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
b = tf.Variable([[0., 1.], [2., 0.], [0., 0.]], dtype=tf.float32)

# Sparse-Dense Matrix Multiplication
# PyTorch: torch.sparse.mm(a, b)
y = tf.sparse.sparse_dense_matmul(a, b)

# Convert result back to dense
# PyTorch: z = y.to_dense()
z = tf.sparse.to_dense(y)
```