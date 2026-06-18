import tensorflow as tf
import numpy as np

def test_tpu_batch_parallel_fusion():
    """
    Adapted from PyTorch issue #162410.
    Tests numerical correctness of tf.compat.v1.tpu.batch_parallel
    with operations involving in-place-like updates (stateful ops) and reductions.
    """
    
    # Check for TPU availability
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError:
        print("TPU not available. This test requires a TPU runtime to execute.")
        return

    # Define the computation function
    # PyTorch: x.copy_(x.flip(1))
    # PyTorch: y = y.sum(dim=1, keepdim=True) + y
    # PyTorch: return x + y
    def computation(x, y):
        # Simulate x.copy_(x.flip(1)) using a Variable to mimic in-place mutation
        # Note: In TF, inputs to batch_parallel are typically Tensors (immutable).
        # We create a Variable to simulate the buffer update logic of the PyTorch bug.
        v_x = tf.Variable(x, trainable=False)
        v_x.assign(tf.reverse(v_x, axis=[1]))
        
        # y = y.sum(dim=1, keepdim=True) + y
        y = tf.reduce_sum(y, axis=1, keepdims=True) + y
        
        # return x + y
        return v_x + y

    # Setup inputs
    # PyTorch: (20, 1024 * 1024). 
    # Adjusted batch size to 16 to be divisible by typical 8-core TPU shards for batch_parallel.
    batch_size = 16
    dim = 1024 * 1024
    
    np_x = np.random.randn(batch_size, dim).astype(np.float32)
    np_y = np.random.randn(batch_size, dim).astype(np.float32)

    with strategy.scope():
        # Run compiled version via batch_parallel
        # This compiles the computation and runs it in parallel on TPU
        act = tf.compat.v1.tpu.batch_parallel(
            computation,
            inputs=[np_x, np_y],
            num_shards=8
        )

        # Run reference version (Eager execution on CPU)
        # We run this on CPU to ensure it's not compiled by XLA/TPU, serving as the ground truth
        with tf.device("/CPU:0"):
            ref_x = tf.constant(np_x)
            ref_y = tf.constant(np_y)
            
            # Replicate the logic exactly for the reference
            v_ref_x = tf.Variable(ref_x, trainable=False)
            v_ref_x.assign(tf.reverse(v_ref_x, axis=[1]))
            ref_y_op = tf.reduce_sum(ref_y, axis=1, keepdims=True) + ref_y
            ref = v_ref_x + ref_y_op

        # Assert close
        # The PyTorch bug was a numerical mismatch due to incorrect fusion.
        # We check if the TPU compiled version matches the CPU eager version.
        act_np = act.numpy()
        ref_np = ref.numpy()
        
        np.testing.assert_allclose(act_np, ref_np, rtol=1e-5, atol=1e-5)
        print("Test passed: TPU batch_parallel output matches eager execution.")

if __name__ == "__main__":
    test_tpu_batch_parallel_fusion()