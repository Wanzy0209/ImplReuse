import tensorflow as tf
import numpy as np

def run_with_dim(dim, device):
    """
    Run batch_parallel with a specific dimension, creating a captured buffer sized by dim.
    This mimics the behavior of run_with_head_count in the PyTorch issue.
    """
    # Create captured buffer that depends on dynamic 'dim'
    # This mimics 'head_scale' in the PyTorch example
    captured_buffer = tf.random.normal([dim], dtype=tf.float16)

    def computation(inputs):
        """
        The computation to be sharded.
        It captures 'captured_buffer' from the outer scope.
        """
        x = inputs[0]
        # Mimic the logic: score * head_scale[head]
        # We multiply the input by the captured buffer to enforce the dependency.
        # Broadcasting will handle the shape alignment.
        return x * captured_buffer

    print(f"  Running with dim={dim}, captured_buffer.shape={captured_buffer.shape}")

    # Run multiple iterations with the same captured_buffer
    for i in range(5):
        # Create inputs. Batch size is fixed, but the feature dimension matches 'dim'
        B = 2
        # Inputs must be a list of Tensors
        inputs = [tf.random.normal([B, dim], dtype=tf.float16)]

        # Call the similar API: tf.compat.v1.tpu.batch_parallel
        # This compiles and runs the computation on TPU cores.
        try:
            outputs = tf.compat.v1.tpu.batch_parallel(
                computation,
                inputs=inputs,
                num_shards=1 # Simplified to 1 shard for general compatibility
            )
            
            # Verify output shape to ensure execution completed
            assert outputs.shape == (B, dim)
            
        except Exception as e:
            print(f"   Failed at iteration {i+1}: {e}")
            raise

    print(f"   Completed {i+1} iterations")


def main():
    # Test with different dimensions - this makes 'dim' a dynamic dimension
    # and the captured buffer changes size with 'dim'
    dims = [4, 8, 4, 16, 4]

    # Note: tf.compat.v1.tpu.batch_parallel requires a TPU context.
    # We attempt to initialize TPU, but the logic focuses on the API call structure.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        device = "/TPU:0"
        print("TPU initialized.")
    except (ValueError, tf.errors.NotFoundError):
        print("TPU not found. This test requires a TPU environment to execute batch_parallel.")
        # We proceed to define the test structure, but execution will likely fail without hardware.
        device = "/CPU:0" 

    print(f"Running batch_parallel with dynamic dimensions")
    print(f"Testing dimensions: {dims}\n")

    for iteration, dim in enumerate(dims, start=1):
        print(f"Iteration {iteration}:")
        run_with_dim(dim, device)


if __name__ == "__main__":
    main()