```python
import tensorflow as tf

# PyTorch: a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
# TensorFlow: Construct SparseTensor manually. To mimic requires_grad, values are a Variable.
a_indices = tf.constant([[0, 0], [0, 2], [1, 1]], dtype=tf.int64)
a_values = tf.Variable([1.0, 2.0, 3.0], dtype=tf.float32)
a = tf.sparse.SparseTensor(a_indices, a_values, [2, 3])

# PyTorch: b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
b = tf.Variable([[0., 1.], [2., 0.], [0., 0.]], dtype=tf.float32)

# PyTorch: y = torch.sparse.mm(a, b)
# TensorFlow: sparse_dense_matmul returns a dense tensor directly.
y = tf.sparse.sparse_dense_matmul(a, b)

# PyTorch: z = y.to_dense()
# TensorFlow: y is already dense, so z is just y.
z = y
```