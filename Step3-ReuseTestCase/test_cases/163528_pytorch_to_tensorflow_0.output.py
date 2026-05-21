import torch
import tensorflow as tf
import numpy as np

class Foo(tf.Module):
    def __init__(self) -> None:
        super().__init__()
        # tf.compat.v1.math.erf is a stateless operation, 
        # but we maintain the class structure to mirror the PyTorch test case.
        pass

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Using the specific API requested: tf.compat.v1.math.erf
        return tf.compat.v1.math.erf(x)

def test_device(device_name, x):
    # Ensure data is on the correct device
    with tf.device(device_name):
        x_device = tf.identity(x)
        
        foo = Foo()
        # tf.function is the TensorFlow equivalent of torch.compile
        foo_compiled = tf.function(foo)

        # warm up
        y_original = foo(x_device)
        y_compiled = foo_compiled(x_device)

        # proper inference
        y_original = foo(x_device)
        y_compiled = foo_compiled(x_device)

        diff = tf.reduce_max(tf.abs(y_original - y_compiled)).numpy()
        print(f'device: {device_name}, diff: {diff}')
        print('original', y_original[:5, :5].numpy())
        print('compiled', y_compiled[:5, :5].numpy())

def main():
    batch_size = 32
    feature_dim = 10
    np.random.seed(42)
    
    # Generate random input data
    x_np = np.random.randn(batch_size, feature_dim).astype(np.float32)
    x = tf.constant(x_np)

    # Test on CPU
    test_device('/CPU:0', x)

    # Test on GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        test_device('/GPU:0', x)
    else:
        print("No GPU available, skipping GPU test.")

if __name__ == '__main__':
    main()