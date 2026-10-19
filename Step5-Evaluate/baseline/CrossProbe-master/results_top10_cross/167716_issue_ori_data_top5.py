```python
import tensorflow as tf

# PyTorch: a = torch.tensor([[1., 0, 2], [0, 3, 0]]).to_sparse().requires_grad_()
# Conversion: Create a SparseTensor. Gradients are tracked via tf.GradientTape, not a property.
indices_a = [[0, 0], [0, 2], [1, 1]]
values_a = [1.0, 2.0, 3.0]
dense_shape_a = [2, 3]
a = tf.SparseTensor(indices=indices_a, values=values_a, dense_shape=dense_shape_a)

# PyTorch: b = torch.tensor([[0, 1.], [2, 0], [0, 0]], requires_grad=True)
# Conversion: Use tf.Variable to mimic requires_grad=True.
b = tf.Variable([[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]], dtype=tf.float32)

# PyTorch: y = torch.sparse.mm(a, b)
# Conversion: Use tf.sparse.sparse_dense_matmul for sparse-dense multiplication.
y = tf.sparse.sparse_dense_matmul(a, b)

# PyTorch: z = y.to_dense()
# Conversion: tf.sparse.sparse_dense_matmul returns a dense tensor by default.
z = y
```