import tensorflow as tf
import numpy as np

# Initialize TPU system required for batch_parallel
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("TPU initialized successfully.")
except ValueError as e:
    print(f"TPU initialization failed: {e}")
    print("This test requires a TPU runtime to execute tf.compat.v1.tpu.batch_parallel.")
    exit(1)

def computation(arg0, arg1):
    """
    Mimics the PyTorch logic:
    t2 = t0.clone()
    t2.fill_diagonal_(t1.item())
    """
    # t2 = t0.clone()
    t2 = tf.identity(arg0)
    
    # t2.fill_diagonal_(t1.item())
    # In TensorFlow, tensors are immutable. We use tf.linalg.set_diag.
    # arg1 is a 0-d tensor (scalar). set_diag expects the diagonal to have shape [..., M].
    # Since t2 is (1, 1), the diagonal must be (1,). We expand arg0 to match.
    diagonal_val = tf.expand_dims(arg1, 0)
    
    return tf.linalg.set_diag(t2, diagonal_val)

# Define inputs
# arg0: size=(1, 1), dtype=float32
arg0 = tf.Variable(np.ones((1, 1), dtype=np.float32))
# arg1: size=(), dtype=float32
arg1 = tf.Variable(5.0, dtype=tf.float32)

if __name__ == '__main__':
    # Run Eager mode (baseline)
    # Note: batch_parallel is specifically for TPU/XLA, so we run the computation function directly
    # to simulate the "Eager" behavior of the original test case.
    out_eager = computation(arg0, arg1)
    print(f"Eager Output: {out_eager.numpy()}")

    # Run Compiled mode (batch_parallel)
    with strategy.scope():
        # batch_parallel expects inputs as a list of tensors
        inputs = [arg0, arg1]
        
        # num_shards=1 to match the single-device context of the original bug report
        out_compiled = tf.compat.v1.tpu.batch_parallel(
            computation,
            inputs=inputs,
            num_shards=1
        )
        
    print(f"Compiled Output: {out_compiled.numpy()}")

    # Verify Forward Pass
    expected = tf.constant([[5.0]], dtype=tf.float32)
    if tf.reduce_all(tf.equal(out_compiled, expected)).numpy():
        print('Forward Pass Success! ')
    else:
        print('Forward Pass Failed! ')
        exit(1)

    # Verify Backward Pass (Gradient Check)
    # The original test case checks backward pass to ensure the graph is fully differentiable.
    with tf.GradientTape() as tape:
        # We run the logic inside the tape. 
        # Since batch_parallel is a high-level API, we wrap the call or the computation.
        # Here we verify gradients for the computation logic itself.
        loss = tf.reduce_sum(computation(arg0, arg1))
    
    grads = tape.gradient(loss, [arg0, arg1])
    
    if grads[0] is not None and grads[1] is not None:
        print('Backward Pass Success! ')
    else:
        print('Backward Pass Failed! ')
        print(f"Gradients: {grads}")