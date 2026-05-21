import torch
import tensorflow as tf
import sys

# Define the computation function
def computation_fn(arg0, arg1, arg2):
    # t0 = arg0 # size=(5699097, 6, 1), dtype=bfloat16
    t1 = tf.sigmoid(arg0) # size=(5699097, 6, 1), dtype=bfloat16
    
    # t2 = arg1 # size=(5699097, 6, 256), dtype=bfloat16
    t3 = tf.sigmoid(arg1) # size=(5699097, 6, 256), dtype=bfloat16
    
    # t4 = arg2 # size=(5699097, 256, 1), dtype=bfloat16
    t5 = tf.exp(arg2) # size=(5699097, 256, 1), dtype=bfloat16
    
    # t6 = torch.baddbmm(t1, t3, t5)
    # Equivalent to: t1 + batch_matmul(t3, t5)
    # t3 shape: (B, 6, 256), t5 shape: (B, 256, 1) -> matmul result: (B, 6, 1)
    # t1 shape: (B, 6, 1)
    # t6 shape: (B, 6, 1)
    t6 = t1 + tf.matmul(t3, t5)
    
    # t7 = t6.reshape((193, 386, 459))
    t7 = tf.reshape(t6, (193, 386, 459))
    
    output = t7
    return output

# Define inputs
# PyTorch: torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# TensorFlow: tf.random.uniform(..., dtype=tf.bfloat16)
arg0 = tf.random.uniform([5699097, 6, 1], dtype=tf.bfloat16)
arg1 = tf.random.uniform([5699097, 6, 256], dtype=tf.bfloat16)
arg2 = tf.random.uniform([5699097, 256, 1], dtype=tf.bfloat16)

if __name__ == '__main__':
    # 1. Run Eager
    try:
        with tf.GradientTape() as tape:
            tape.watch([arg0, arg1, arg2])
            out_eager = computation_fn(arg0, arg1, arg2)
        grads = tape.gradient(out_eager, [arg0, arg1, arg2])
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')

    # 2. Run via tf.compat.v1.tpu.rewrite
    # Note: This requires a TPU environment to execute fully.
    # We wrap the TPU initialization and execution in a try-except block
    # to handle environments without TPUs gracefully.
    try:
        # Initialize TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            # tf.compat.v1.tpu.rewrite compiles the computation for TPU.
            # We wrap the call in a tf.function to ensure it is compiled/traced.
            @tf.function
            def run_compiled():
                # The rewrite function takes the computation and the list of inputs.
                return tf.compat.v1.tpu.rewrite(computation_fn, [arg0, arg1, arg2])

            out_compiled = run_compiled()
            
            # Check gradients (backward pass equivalent)
            with tf.GradientTape() as tape:
                tape.watch([arg0, arg1, arg2])
                out_compiled_grad = run_compiled()
            grads_compiled = tape.gradient(out_compiled_grad, [arg0, arg1, arg2])

        print('Compile (TPU Rewrite) Success! ')
    except Exception as e:
        # This block catches errors if no TPU is found or if OOM occurs on TPU.
        print(f'Compile (TPU Rewrite) Failed or Skipped (No TPU?): {e}')