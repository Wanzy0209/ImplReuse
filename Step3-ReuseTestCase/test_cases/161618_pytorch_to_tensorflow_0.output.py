import torch
import tensorflow as tf

# Define the specific tensor shapes from the original bug report
m = 20120
k = 1536
n = 512

# Define the computation function equivalent to the PyTorch lambda:
# lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)
# torch.addmm(a, mat1, mat2) computes a + mat1 @ mat2
def computation_fn(a, mat1, mat2):
    return tf.add(a, tf.matmul(mat1, mat2))

def main():
    # Initialize TPU system (required for tf.compat.v1.tpu.rewrite)
    # Note: This code block requires a TPU environment to execute fully.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        
        print("TPU system initialized.")

        # Create input tensors
        # PyTorch: torch.randn -> TensorFlow: tf.random.normal
        # PyTorch: .cuda() -> TPU handles placement via strategy/rewrite
        a = tf.random.normal((m, n))
        mat1 = tf.random.normal((m, k))
        mat2 = tf.random.normal((k, n))

        # Execute the compilation and run using the similar API
        # This mirrors the torch.compile(f, dynamic=False) call
        with strategy.scope():
            # tf.compat.v1.tpu.rewrite compiles the computation for TPU
            # and executes it with the provided inputs.
            result = tf.compat.v1.tpu.rewrite(computation_fn, [a, mat1, mat2])

        # Verify the behavior
        # The API returns a list of tensors corresponding to the function output
        assert isinstance(result, list), "Result should be a list of tensors"
        assert len(result) == 1, "Expected one output tensor"
        
        output_tensor = result[0]
        assert output_tensor.shape == (m, n), f"Expected shape ({m}, {n}), got {output_tensor.shape}"
        
        print("Test passed: tf.compat.v1.tpu.rewrite executed successfully with the specified workload.")

    except tf.errors.NotFoundError:
        print("TPU hardware not found. Skipping test execution as it requires a TPU runtime.")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    main()