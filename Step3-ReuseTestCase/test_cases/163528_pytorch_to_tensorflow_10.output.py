import torch
import tensorflow as tf
import numpy as np

class Foo(tf.Module):
    """
    TensorFlow module wrapping the target API.
    Analogous to the PyTorch nn.Module in the original bug report.
    """
    def __init__(self) -> None:
        super().__init__()
        # Note: tf.compat.v1.math.cosh operates on a single input tensor,
        # unlike torch.searchsorted which takes boundaries and values.
        # We maintain the class structure to mirror the original test setup.

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        # Eager execution of the target API
        return tf.compat.v1.math.cosh(x)

class FooCompiled(tf.Module):
    """
    TensorFlow module using tf.function to simulate compilation.
    Analogous to torch.compile in the original bug report.
    """
    def __init__(self) -> None:
        super().__init__()

    @tf.function
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        # Graph execution (compiled) of the target API
        return tf.compat.v1.math.cosh(x)

def test_device(device_name, x):
    """
    Tests the API on a specific device (CPU or GPU), comparing
    eager execution against graph (compiled) execution.
    """
    # Ensure operations run on the specified device
    with tf.device(device_name):
        foo = Foo()
        foo_compiled = FooCompiled()

        # warm up
        y_eager = foo.forward(x)
        y_compiled = foo_compiled.forward(x)

        # proper inference
        y_eager = foo.forward(x)
        y_compiled = foo_compiled.forward(x)

        # Calculate difference
        diff = tf.reduce_max(tf.abs(y_eager - y_compiled)).numpy()
        
        print(f'device: {device_name}, diff: {diff}')
        print('eager', y_eager.numpy()[:5, :5])
        print('compiled', y_compiled.numpy()[:5, :5])

        # Assertion to verify behavior consistency
        # Using a small epsilon to account for floating point precision differences
        assert diff < 1e-5, f"Significant difference found on {device_name} between eager and compiled execution."

def main():
    batch_size = 32
    feature_dim = 10
    
    # Setup input data
    np.random.seed(42)
    x_np = np.random.randn(batch_size, feature_dim).astype(np.float32)
    x_tensor = tf.constant(x_np)

    # Test CPU
    print("Testing on CPU...")
    test_device('/CPU:0', x_tensor)

    # Test GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Testing on GPU...")
        test_device('/GPU:0', x_tensor)
    else:
        print("No GPU found, skipping GPU test.")

if __name__ == '__main__':
    main()