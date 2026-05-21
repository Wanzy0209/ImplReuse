import torch
import tensorflow as tf
import numpy as np

class Foo(tf.Module):
    def __init__(
        self,
        quantiles: tf.Tensor,
    ) -> None:
        super().__init__()
        # We keep the parameter structure to mimic the original model setup,
        # even though negative is a unary operation.
        self.q = tf.Variable(quantiles, trainable=False)

    @tf.function
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Adapted from torch.searchsorted(self.q, x.T).T
        # to use the target API: tf.compat.v1.math.negative
        return tf.compat.v1.math.negative(x)

def test_device(device_name, x, quantiles):
    # Reset graph/state for clean run per device
    tf.keras.backend.clear_session()
    
    with tf.device(device_name):
        foo = Foo(quantiles)
        
        # 1. Eager Execution (Baseline)
        # In TF, calling a tf.function or module directly executes eagerly if not decorated/traced yet,
        # but here we ensure we get the raw result first.
        # Note: Since __call__ is decorated with @tf.function, the first call traces it.
        # To strictly mimic "eager vs compiled", we can disable function for the first part or just rely on XLA.
        # However, the most relevant comparison for "torch.compile" bugs in TF is Eager vs XLA (jit_compile=True).
        
        # Let's get an eager result by creating a non-decorated version or just running the op.
        y_eager = tf.compat.v1.math.negative(x)

        # 2. Compiled Execution (XLA)
        # This is the TensorFlow equivalent to torch.compile(..., fullgraph=True)
        foo_compiled = tf.function(foo, jit_compile=True)
        
        # Warm up (tracing)
        _ = foo_compiled(x)
        
        # Proper inference
        y_compiled = foo_compiled(x)

        diff = tf.reduce_max(tf.abs(y_eager - y_compiled))
        print(f'device: {device_name}, diff: {diff.numpy()}')
        print('eager', y_eager[:5, :5].numpy())
        print('compiled', y_compiled[:5, :5].numpy())
        
        # Assertion to catch potential bugs
        np.testing.assert_allclose(y_eager.numpy(), y_compiled.numpy(), rtol=1e-5)

def main():
    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    
    # Setup data
    np.random.seed(42)
    x_np = np.random.randn(batch_size, feature_dim).astype(np.float32)
    quantiles_np = np.random.randn(quantile_size, feature_dim).astype(np.float32)
    quantiles_np = np.sort(quantiles_np, axis=0)
    
    x = tf.constant(x_np)
    quantiles = tf.constant(quantiles_np)

    # Test CPU
    print("--- Testing CPU ---")
    test_device('/CPU:0', x, quantiles)

    # Test GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("--- Testing GPU ---")
        test_device('/GPU:0', x, quantiles)
    else:
        print("No GPU found, skipping GPU test.")

if __name__ == '__main__':
    main()