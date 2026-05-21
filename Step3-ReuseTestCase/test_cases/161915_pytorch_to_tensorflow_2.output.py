import torch
import tensorflow as tf

# Adaptation of the PyTorch test case for tf.io.serialize_sparse
# Original: Create NestedTensor with tensors of size 3 and 5, then call share_memory_()
# Adapted: Create SparseTensor representing rows of size 3 and 5, then call serialize_sparse()

# 1. Create the irregular data structure (SparseTensor)
# Row 0 has 3 elements, Row 1 has 5 elements
indices = tf.constant([
    [0, 0], [0, 1], [0, 2],
    [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]
], dtype=tf.int64)

# Generate random values (equivalent to torch.randn)
values = tf.random.normal([8], dtype=tf.float32)

# Define the shape (2 rows, max 5 columns)
shape = tf.constant([2, 5], dtype=tf.int64)

# Construct the SparseTensor
st = tf.SparseTensor(indices, values, shape)
print(f"Input SparseTensor: {st}")

# 2. Call the similar API (tf.io.serialize_sparse)
# This corresponds to the operation on the irregular data structure.
# We verify that this operation completes without a segmentation fault.
try:
    serialized_tensor = tf.io.serialize_sparse(st)
    
    # Verify the output structure
    # serialize_sparse returns a 3-vector (1-D Tensor) of strings
    assert serialized_tensor.shape == (3,), f"Expected shape (3,), got {serialized_tensor.shape}"
    assert serialized_tensor.dtype == tf.string, f"Expected dtype string, got {serialized_tensor.dtype}"
    
    print(f"Serialization successful. Output: {serialized_tensor}")
except Exception as e:
    print(f"Test failed with exception: {e}")
    raise