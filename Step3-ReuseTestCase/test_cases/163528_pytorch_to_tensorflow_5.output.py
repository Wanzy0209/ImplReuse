import torch
import tensorflow as tf
import numpy as np

# Define the model using the target API: tf.keras.ops.negative
class Foo(tf.Module):
    def __init__(self) -> None:
        super().__init__()
        # The original PyTorch code used a parameter 'quantiles' for searchsorted.
        # For tf.keras.ops.negative, we don't need a separate parameter 
        # to perform the operation on 'x', but we keep the class structure.
        self.dummy_var = tf.Variable(0.0, trainable=False)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Original: torch.searchsorted(self.q, x.T).T
        # Adapted: tf.keras.ops.negative(x)
        return tf.keras.ops.negative(x)

def test_device(device_name, x):
    # Set device context
    with tf.device(device_name):
        foo = Foo()
        
        # TensorFlow equivalent of torch.compile is tf.function.
        # This traces the function and creates a compiled graph.
        # fullgraph=True in PyTorch implies compiling the whole model, 
        # which is the default behavior of tf.function for a simple module.
        foo_compiled = tf.function(foo)

        # warm up
        y_original = foo(x)
        y_compiled = foo_compiled(x)

        # proper inference
        y_original = foo(x)
        y_compiled = foo_compiled(x)

        # Calculate difference
        diff = tf.reduce_max(tf.abs(y_original - y_compiled)).numpy()
        print(f'device: {device_name}, diff: {diff}')
        print('original', y_original[:5, :5].numpy())
        print('compiled', y_compiled[:5, :5].numpy())
        
        # Assert that the difference is negligible (checking for the bug)
        assert diff < 1e-6, f"Discrepancy found on {device_name}"

def main():
    batch_size = 32
    feature_dim = 10
    
    # Set seed for reproducibility
    tf.random.set_seed(42)
    x = tf.random.normal((batch_size, feature_dim), dtype=tf.float32)

    # Determine available devices
    devices = ['/CPU:0']
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        devices.append('/GPU:0')

    for device in devices:
        test_device(device, x)

if __name__ == '__main__':
    main()