import torch
import tensorflow as tf
import numpy as np

def test_tpu_batch_parallel_addmm():
    """
    Adapts the PyTorch Inductor/Triton bug scenario to TensorFlow's TPU API.
    
    Original Bug Context:
    - PyTorch Inductor failed compiling `torch.addmm(a, mat1, mat2)` with specific shapes.
    - Shapes: m=20120, k=1536, n=512.
    - Operation: Matrix multiplication (mat1 @ mat2) added to a bias tensor (a).
    
    Adaptation Logic:
    - Use `tf.compat.v1.tpu.batch_parallel` to shard the computation.
    - The operation `addmm` maps to `tf.add(a, tf.matmul(mat1, mat2))`.
    - Sharding is performed along the batch dimension (dimension 0).
    - `mat2` (K x N) is treated as a weight matrix and broadcast to all shards,
      while `a` (M x N) and `mat1` (M x K) are sharded along M.
    """
    
    # Initialize TPU system
    # Note: This requires a TPU environment (e.g., Colab with TPU runtime)
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        print("TPU not found, skipping test.")
        return

    # Dimensions from the original bug report
    m = 20120
    k = 1536
    n = 512

    with strategy.scope():
        # Create input tensors matching the original shapes and dtypes (float32)
        a = tf.random.normal((m, n), dtype=tf.float32)
        mat1 = tf.random.normal((m, k), dtype=tf.float32)
        mat2 = tf.random.normal((k, n), dtype=tf.float32)

        # Define the computation function that will run on each shard.
        # This corresponds to the lambda function in the PyTorch script.
        def computation(a_shard, mat1_shard):
            # mat2 is captured from the outer scope. In TPU sharding, variables/tensors
            # not in the 'inputs' list are typically broadcast to all cores.
            # This preserves the logic of mat2 being the full (K, N) matrix.
            return tf.add(a_shard, tf.matmul(mat1_shard, mat2))

        # Execute the computation using batch_parallel
        # inputs=[a, mat1] are split along dimension 0.
        result = tf.compat.v1.tpu.batch_parallel(
            computation,
            inputs=[a, mat1],
            num_shards=strategy.num_replicas_in_sync
        )

        # Verification: Calculate the expected result on the host/CPU
        # to ensure the sharded TPU execution is mathematically correct.
        expected = tf.add(a, tf.matmul(mat1, mat2))

        # Assert shape consistency
        assert result.shape == expected.shape, \
            f"Shape mismatch: got {result.shape}, expected {expected.shape}"

        # Assert numerical consistency
        # We use a tolerance suitable for float32 operations across different devices
        tf.debugging.assert_near(result, expected, rtol=1e-5, atol=1e-5, 
                                 message="TPU batch_parallel result diverges from expected")

        print("Test passed: tf.compat.v1.tpu.batch_parallel handled the addmm workload correctly.")

if __name__ == "__main__":
    test_tpu_batch_parallel_addmm()