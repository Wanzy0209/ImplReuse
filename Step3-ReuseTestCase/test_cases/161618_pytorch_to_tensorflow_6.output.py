import torch
import tensorflow as tf

# Dimensions from the original bug report
m = 20120
k = 1536
n = 512

# Create tensors (TensorFlow defaults to float32, similar to PyTorch's randn)
a = tf.random.normal((m, n))
mat1 = tf.random.normal((m, k))
mat2 = tf.random.normal((k, n))

# Define the function equivalent to torch.addmm(a, mat1, mat2)
# which computes a + mat1 @ mat2
def compute_addmm(a, mat1, mat2):
    return a + tf.matmul(mat1, mat2)

# Use the target API: tf.compat.v1.name_scope
# This context manager is used to provide a prefix for operations created within it.
# We adapt the original logic by wrapping the compilation and execution in this scope.
with tf.compat.v1.name_scope("autotune_scope"):
    # In TensorFlow, tf.function is the equivalent of torch.compile for graph optimization
    @tf.function
    def compiled_func(a, mat1, mat2):
        return compute_addmm(a, mat1, mat2)

    # Execute the compiled function
    result = compiled_func(a, mat1, mat2)

# Verify the result shape matches the expected output
assert result.shape == (m, n), f"Expected shape {(m, n)}, but got {result.shape}"

# Verify that the name scope was applied to the operations in the graph
# This checks the specific functionality of the API under test
concrete_func = compiled_func.get_concrete_function(a, mat1, mat2)
graph = concrete_func.graph
ops = [op.name for op in graph.get_operations()]
assert any("autotune_scope" in op_name for op_name in ops), \
    "Name scope was not applied to the graph operations."

print("Test passed.")