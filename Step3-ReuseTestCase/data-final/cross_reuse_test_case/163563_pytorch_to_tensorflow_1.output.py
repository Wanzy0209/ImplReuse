import torch
import tensorflow as tf
import sys

def run_test():
    # Initialize TPU
    print("Initializing TPU...")
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"TPU initialized: {resolver.master()}")
    except ValueError as e:
        print(f"TPU not found or initialization failed: {e}")
        print("This test requires a TPU runtime to execute tf.compat.v1.tpu.batch_parallel.")
        return

    # Define the computation function
    # Translating torch operations to tf operations
    def computation_fn(arg0, arg1, arg2):
        # t0 = arg0
        # t1 = torch.sigmoid(t0)
        t1 = tf.sigmoid(arg0)
        
        # t2 = arg1
        # t3 = torch.sigmoid(t2)
        t3 = tf.sigmoid(arg1)
        
        # t4 = arg2
        # t5 = torch.exp(t4)
        t5 = tf.exp(arg2)
        
        # t6 = torch.baddbmm(t1, t3, t5)
        # baddbmm(input, batch1, batch2) -> input + batch1 @ batch2
        # Shapes: t1(B, 6, 1), t3(B, 6, 256), t5(B, 256, 1)
        # matmul: t3 @ t5 -> (B, 6, 1)
        # add: t1 + result -> (B, 6, 1)
        matmul_res = tf.matmul(t3, t5)
        t6 = t1 + matmul_res
        
        # t7 = t6.reshape((193, 386, 459))
        # Note: We use -1 for the last dimension to ensure the reshape is valid 
        # regardless of exact batch size divisibility, preserving the logic of reshaping.
        t7 = tf.reshape(t6, (193, 386, -1))
        
        return t7

    # Define inputs
    # Original size: 5699097. 
    # We use the original size to preserve the OOM reproduction logic.
    # Note: batch_parallel requires the batch dimension to be divisible by num_shards.
    # We will use num_shards=1 to strictly adhere to the input sizes provided in the bug report
    # while still utilizing the batch_parallel API path.
    batch_size = 5699097
    
    # Using tf.random.uniform to match torch.rand
    # dtype bfloat16 to match torch.bfloat16
    arg0 = tf.random.uniform([batch_size, 6, 1], dtype=tf.bfloat16)
    arg1 = tf.random.uniform([batch_size, 6, 256], dtype=tf.bfloat16)
    arg2 = tf.random.uniform([batch_size, 256, 1], dtype=tf.bfloat16)

    # 1. Run Eager (Standard TF execution)
    print("Running Eager execution...")
    try:
        with tf.GradientTape() as tape:
            tape.watch([arg0, arg1, arg2])
            out_eager = computation_fn(arg0, arg1, arg2)
            loss_eager = tf.reduce_sum(out_eager)
        grads_eager = tape.gradient(loss_eager, [arg0, arg1, arg2])
        print('Eager Success! ')
    except tf.errors.ResourceExhaustedError:
        print('Eager OOM! ')
        return

    # 2. Run Batch Parallel (TPU execution)
    # We use num_shards=1 to accommodate the specific prime batch size from the bug report
    # while still invoking the batch_parallel compilation path.
    print("Running Batch Parallel...")
    
    # We need to execute this on the TPU device
    with tf.device("/device:TPU:0"):
        # Move inputs to TPU
        tpu_arg0 = tf.identity(arg0)
        tpu_arg1 = tf.identity(arg1)
        tpu_arg2 = tf.identity(arg2)

        @tf.function(experimental_compile=True) # Force XLA compilation
        def run_batch_parallel():
            return tf.compat.v1.tpu.batch_parallel(
                computation_fn,
                [tpu_arg0, tpu_arg1, tpu_arg2],
                num_shards=1
            )

        try:
            out_parallel = run_batch_parallel()
            
            # Check gradients
            with tf.GradientTape() as tape:
                tape.watch([tpu_arg0, tpu_arg1, tpu_arg2])
                out_parallel_loss = tf.reduce_sum(out_parallel)
            grads_parallel = tape.gradient(out_parallel_loss, [tpu_arg0, tpu_arg1, tpu_arg2])
            
            print('Batch Parallel Success! ')
            
            # Verify shapes match
            if out_eager.shape != out_parallel.shape:
                print(f"Shape Mismatch: Eager {out_eager.shape} vs Parallel {out_parallel.shape}")
            else:
                print("Shape Match Verified.")

        except tf.errors.ResourceExhaustedError:
            print('Batch Parallel OOM!  (Bug Reproduced)')
        except Exception as e:
            print(f"Batch Parallel Error: {e}")

if __name__ == '__main__':
    run_test()