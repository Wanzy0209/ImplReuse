import torch
import tensorflow as tf
import numpy as np

class Foo(tf.Module):
    def __init__(self) -> None:
        super().__init__()
        # No parameters required for conjugate, unlike searchsorted
        pass

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Using the target API: tf.experimental.numpy.conjugate
        return tf.experimental.numpy.conjugate(x)

def test_device(device_name, x):
    # Place tensors and operations on the specified device
    with tf.device(device_name):
        foo = Foo()
        # tf.function is the TensorFlow equivalent of torch.compile
        foo_compiled = tf.function(foo)

        # warm up
        y_original = foo(x)
        y_compiled = foo_compiled(x)

        # proper inference
        y_original = foo(x)
        y_compiled = foo_compiled(x)

        diff = tf.reduce_max(tf.abs(y_original - y_compiled)).numpy()
        print(f'device: {device_name}, diff: {diff}')
        print('original', y_original[:5, :5].numpy())
        print('compiled', y_compiled[:5, :5].numpy())
        
        # Assert to verify behavior matches between eager and compiled modes
        assert diff < 1e-6, f"Discrepancy found on {device_name}"

def main():
    batch_size = 32
    feature_dim = 10
    
    # Generate complex numbers to make conjugate meaningful
    # (conjugate of real numbers is a no-op)
    tf.random.set_seed(42)
    real_part = tf.random.normal((batch_size, feature_dim))
    imag_part = tf.random.normal((batch_size, feature_dim))
    x = tf.complex(real_part, imag_part)

    # Test CPU
    test_device('/CPU:0', x)

    # Test GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        test_device('/GPU:0', x)
    else:
        print("No GPU found, skipping GPU test.")

if __name__ == '__main__':
    main()