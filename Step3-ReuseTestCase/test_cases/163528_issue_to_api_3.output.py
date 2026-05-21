import torch
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np

class TF_Foo(tf.Module):
    def __init__(
        self,
        base_tensor: tf.Tensor,
    ) -> None:
        super().__init__()
        # Mimic the parameter storage from the PyTorch example
        self.base = tf.Variable(base_tensor, trainable=False)

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        # Use the similar API (tf.experimental.numpy.append) 
        # preserving the structural logic of the original bug (transposes)
        return tnp.append(self.base, x.T, axis=0).T

    # The TensorFlow equivalent of torch.compile is tf.function
    @tf.function
    def forward_compiled(self, x: tf.Tensor) -> tf.Tensor:
        return tnp.append(self.base, x.T, axis=0).T

def test_device(device, x, base):
    with tf.device(device):
        x_tf = tf.convert_to_tensor(x)
        base_tf = tf.convert_to_tensor(base)
        
        foo = TF_Foo(base_tf)

        # warm up
        y_original = foo.forward(x_tf)
        y_compiled = foo.forward_compiled(x_tf)

        # proper inference
        y_original = foo.forward(x_tf)
        y_compiled = foo.forward_compiled(x_tf)

        diff = tf.reduce_max(tf.abs(y_original - y_compiled)).numpy()
        print(f'device: {device}, diff: {diff}')
        print('original', y_original[:5, :5].numpy())
        print('compiled', y_compiled[:5, :5].numpy())
        
        # Assert that the compiled version matches the eager version
        assert diff < 1e-5, f"Diff too high on {device}: {diff}"

def main():
    batch_size = 32
    feature_dim = 10
    base_size = 100
    np.random.seed(42)
    
    x = np.random.randn(batch_size, feature_dim).astype(np.float32)
    base = np.random.randn(base_size, feature_dim).astype(np.float32)
    
    # Test CPU
    test_device('/CPU:0', x, base)

    # Test GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        test_device('/GPU:0', x, base)
    else:
        print("GPU not available, skipping GPU test.")

if __name__ == '__main__':
    main()