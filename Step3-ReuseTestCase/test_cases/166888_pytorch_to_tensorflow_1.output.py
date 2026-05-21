import torch
import tensorflow as tf
import numpy as np

def test_batch_parallel_with_scalar_arg():
    """
    Adapted test case for tf.compat.v1.tpu.batch_parallel based on 
    PyTorch issue #166888 (NameError with .item() on float tensor arg).
    
    This test verifies that batch_parallel handles a scalar tensor argument
    correctly when used in a computation (clamping), similar to the 
    original PyTorch scenario.
    """
    
    # Initialize TPU system (Required for batch_parallel)
    # Note: This requires a TPU environment to run.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)

    # Define inputs
    # x: A batch tensor
    x = tf.random.normal((10, 20, 30))
    # max_val: A scalar float tensor (equivalent to torch.tensor(5.0))
    max_val = tf.constant(5.0)

    # Define the computation function
    # Original PyTorch: def f(x, max_val): return torch.clamp(x, 0, max_val.item())
    # TensorFlow adaptation: 
    # We pass max_val as a tensor argument. Unlike PyTorch's .item() which extracts 
    # a Python float (causing the compilation bug in Inductor), TensorFlow's XLA 
    # compiler handles tensor arguments directly in the graph.
    def computation(x, max_val):
        # tf.minimum(tf.maximum(x, 0), max_val) is equivalent to torch.clamp(x, 0, max_val)
        return tf.minimum(tf.maximum(x, 0.0), max_val)

    # Run batch_parallel
    # We set num_shards=1 to ensure the scalar tensor 'max_val' can be passed as an input
    # without causing a sharding error (since you cannot shard a scalar along dimension 0).
    # This preserves the function signature f(x, max_val) from the original bug report.
    outputs = tf.compat.v1.tpu.batch_parallel(
        computation, 
        inputs=[x, max_val], 
        num_shards=1
    )

    # batch_parallel returns a list of outputs (one per shard). 
    # With num_shards=1, we expect a single output tensor.
    result = outputs[0]

    # Assertions to verify correct behavior
    # 1. Check shape
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} != {x.shape}"
    
    # 2. Check clamping logic (values should be <= max_val and >= 0)
    assert tf.reduce_all(result <= max_val).numpy(), "Values exceed max_val"
    assert tf.reduce_all(result >= 0.0).numpy(), "Values are below 0"

    print("Test passed: batch_parallel handled scalar tensor argument correctly.")

if __name__ == "__main__":
    test_batch_parallel_with_scalar_arg()