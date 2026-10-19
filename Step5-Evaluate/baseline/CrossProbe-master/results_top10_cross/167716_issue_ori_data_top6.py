```python
import tensorflow as tf

# Create tensor a and convert to sparse
# PyTorch: torch.tensor(...).to_sparse().requires_grad_()
# Note: In TensorFlow, gradient tracking is handled by tf.GradientTape.
a = tf.sparse.from_dense(tf.constant([[1., 0, 2], [0, 3, 0]]))

# Create tensor b
# PyTorch: torch.tensor(..., requires_grad=True)
# Note: Using tf.Variable to indicate trainability/gradient tracking intent.
b = tf.Variable([[0, 1.], [2, 0], [0, 0]])

# Sparse matrix multiplication (Sparse x Dense)
# PyTorch: torch.sparse.mm
# Note: tf.sparse.sparse_dense_matmul returns a dense tensor.
y = tf.sparse.sparse_dense_matmul(a, b)

# Convert result to dense
# PyTorch: y.to_dense()
# Note: y is already dense in TensorFlow, so we simply assign it.
z = y
```