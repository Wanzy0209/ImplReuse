import torch
import tensorflow as tf
import numpy as np

class NegativeModel(tf.keras.Model):
    """
    TensorFlow equivalent of the PyTorch Foo module, adapted to use 
    the similar API: tf.math.negative.
    """
    def __init__(self):
        super(NegativeModel, self).__init__()
        # Maintaining a parameter structure similar to the original PyTorch module
        # to ensure the model state handling is consistent.
        self.dummy_param = tf.Variable(initial_value=1.0, trainable=False, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Using the similar API: tf.math.negative
        return tf.math.negative(x)

def test_device(device_name, x):
    """
    Tests the model on the specified device, comparing eager execution 
    against compiled execution (tf.function).
    """
    print(f'--- Testing device: {device_name} ---')
    
    with tf.device(device_name):
        model = NegativeModel()
        
        # 1. Eager execution (Original behavior)
        y_eager = model(x)

        # 2. Compiled execution (tf.function is analogous to torch.compile)
        # We compile the call method specifically
        compiled_call = tf.function(model.call)
        
        # Warm up
        with tf.no_grad(): # Note: TF doesn't use no_grad context, but gradients aren't calculated here
            _ = compiled_call(x)

        # Proper inference
        with tf.no_grad():
            y_compiled = compiled_call(x)

        # Calculate difference
        diff = tf.reduce_max(tf.abs(y_eager - y_compiled))
        print(f'Max difference (Eager vs Compiled): {diff.numpy()}')
        
        # Print samples for visual verification
        print('Eager sample:', y_eager[0, :5].numpy())
        print('Compiled sample:', y_compiled[0, :5].numpy())
        print('')

def main():
    batch_size = 32
    feature_dim = 10
    
    # Setup data
    np.random.seed(42)
    x = tf.constant(np.random.randn(batch_size, feature_dim), dtype=tf.float32)

    # Test CPU
    test_device('CPU', x)

    # Test GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        test_device('GPU', x)
    else:
        print("No GPU available, skipping GPU test.")

if __name__ == '__main__':
    main()