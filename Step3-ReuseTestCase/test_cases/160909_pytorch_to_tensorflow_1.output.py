import torch
import tensorflow as tf
import sys

def test_batch_parallel_device_handling():
    """
    Adapted from PyTorch Issue 160909.
    
    Original Issue: torch.compile with PrivateUse1 backend sees tensors on "meta" device,
    causing a RuntimeError during operations like repeat_interleave.
    
    This test adapts the logic to TensorFlow's tf.compat.v1.tpu.batch_parallel.
    It verifies if the TPU compilation/parallelization API handles operations
    (specifically tf.repeat, analogous to repeat_interleave) correctly without
    device mismatch errors during the graph compilation phase.
    """
    
    # 1. Define the computation (analogous to the Model in PyTorch)
    # The PyTorch error log mentioned 'repeat_interleave'. We use tf.repeat here.
    def computation_fn(inputs):
        x = inputs[0]
        # Perform the operation that triggered the device mismatch in PyTorch
        return tf.repeat(x, repeats=2, axis=1)

    # 2. Prepare inputs
    # PyTorch: data.to('PrivateUse1')
    # In TF, we define the data. batch_parallel will handle sharding and placement.
    # Shape (1, 8, 3, 128) matches the PyTorch bug report.
    data = tf.random.normal([1, 8, 3, 128])

    # 3. Setup TPU environment (Analogous to setting up the PrivateUse1 backend)
    # Note: This requires a TPU environment to run fully, but we wrap it to be safe.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except (ValueError, tf.errors.NotFoundError) as e:
        print(f"Skipping test: TPU not available. This is expected in non-TPU environments. Error: {e}")
        return

    # 4. Execute the parallel compilation and run
    # PyTorch: compiled = torch.compile(model, backend=my_backend)
    # PyTorch: result = compiled(data)
    # TF: batch_parallel compiles the computation for TPU and executes it.
    
    with strategy.scope():
        try:
            result = tf.compat.v1.tpu.batch_parallel(
                computation_fn,
                inputs=[data],
                num_shards=strategy.num_replicas_in_sync
            )
            
            # 5. Verify the result
            # If the bug existed in TF (e.g., device mismatch during compilation),
            # the execution above would fail before reaching this assertion.
            # Input shape: (1, 8, 3, 128) -> Repeat axis 1 by 2 -> (1, 16, 3, 128)
            expected_shape = [1, 16, 3, 128]
            
            # Check if the operation ran successfully and shape is correct
            assert result.shape.as_list() == expected_shape, \
                f"Shape mismatch. Expected {expected_shape}, got {result.shape.as_list()}"
            
            print("Test passed: batch_parallel handled the operation and device placement correctly.")

        except RuntimeError as e:
            # Catching potential runtime errors similar to the PyTorch bug:
            # "RuntimeError: ... storage is not on the custom device"
            print(f"Test failed with RuntimeError: {e}")
            raise
        except Exception as e:
            print(f"Test failed with unexpected error: {e}")
            raise

if __name__ == "__main__":
    test_batch_parallel_device_handling()