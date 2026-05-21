import torch
import tensorflow as tf

def test_name_scope_operations():
    """
    Adapted test case for tf.keras.backend.name_scope.
    
    The original bug report involved torch.compile failing with a NameError 
    when combining operations (sum, mean, cumsum) with combo_kernels enabled.
    
    Since tf.keras.backend.name_scope is a context manager for naming operations
    in a graph rather than a JIT compiler, we adapt the test to verify that
    the core operations (sum, mean, cumsum) execute correctly within the scope.
    """
    
    # Define inputs similar to the original PyTorch repro
    # Original: torch.rand(16, 128, device="cuda")
    # TF: tf.random.uniform (TensorFlow handles device placement automatically)
    inps = [
        tf.random.uniform((16, 128)),
        tf.random.uniform((32, 128)),
        tf.random.uniform((32, 256)),
    ]

    # Use the similar API: tf.keras.backend.name_scope
    # This acts as the context wrapper, analogous to the compilation context in the original.
    with tf.keras.backend.name_scope("combo_ops_scope"):
        x, y, z = inps
        
        # Perform the operations from the original bug report
        # x.sum(1) -> tf.reduce_sum(x, axis=1)
        # y.mean(1) -> tf.reduce_mean(y, axis=1)
        # z.cumsum(1) -> tf.cumsum(z, axis=1)
        out_sum = tf.reduce_sum(x, axis=1)
        out_mean = tf.reduce_mean(y, axis=1)
        out_cumsum = tf.cumsum(z, axis=1)

        # Verify the operations executed successfully and shapes are correct
        assert out_sum.shape == (16,), f"Expected shape (16,), got {out_sum.shape}"
        assert out_mean.shape == (32,), f"Expected shape (32,), got {out_mean.shape}"
        assert out_cumsum.shape == (32, 256), f"Expected shape (32, 256), got {out_cumsum.shape}"

    print("Test passed: Operations executed successfully within name_scope.")

if __name__ == "__main__":
    test_name_scope_operations()