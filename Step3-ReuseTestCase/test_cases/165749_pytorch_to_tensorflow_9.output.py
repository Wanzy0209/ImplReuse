import torch
import tensorflow as tf
import numpy as np

# Custom layer to mimic torch.nn.utils.parametrizations.weight_norm
# PyTorch weight_norm reparametrizes the weight tensor w as:
# w = g * (v / ||v||)
# where v is the direction tensor and g is the magnitude.
class WeightNormConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        # Input shape for NCHW: (batch, channels, height, width)
        # Kernel shape for NCHW: (kernel_h, kernel_w, in_channels, out_channels)
        in_channels = input_shape[1]
        
        # Initialize v (direction)
        v_init = tf.random_normal_initializer()
        self.v = self.add_weight(
            name="v",
            shape=(self.kernel_size, self.kernel_size, in_channels, self.filters),
            initializer=v_init,
            trainable=True
        )
        
        # Initialize g (magnitude)
        g_init = tf.ones_initializer()
        self.g = self.add_weight(
            name="g",
            shape=(self.filters,),
            initializer=g_init,
            trainable=True
        )
        
        super().build(input_shape)

    def call(self, inputs):
        # Calculate normalized v
        # Normalize over all dimensions except the output channel (last axis)
        v_norm = tf.nn.l2_normalize(self.v, axis=[0, 1, 2], epsilon=1e-12)
        
        # Reconstruct weight w
        w = self.g * v_norm
        
        # Perform convolution
        # Using data_format='NCHW' to match PyTorch's default
        return tf.nn.conv2d(
            inputs, 
            w, 
            strides=[1, 1, 1, 1], 
            padding='VALID', 
            data_format='NCHW'
        )

def test_tf_name_scope_conv():
    """
    Adapts the PyTorch bug reproducer to TensorFlow using tf.name_scope.
    The original bug involves torch.compile, weight_norm, and Conv2d with d > 64.
    Here we verify the execution within a tf.name_scope context.
    """
    d = 65  # Trigger condition from the bug report (d > 64)
    
    # Create input tensor (NCHW format to match PyTorch)
    x = tf.random.normal((1, 2, 32, 32))
    
    # The original API torch.compile wraps the model.
    # The similar API tf.name_scope wraps the definition/execution context.
    with tf.name_scope("weight_norm_conv_scope"):
        # Instantiate the custom model
        model = WeightNormConv2D(filters=d, kernel_size=2)
        
        # Optimizer
        opt = tf.keras.optimizers.SGD()
        
        # Training loop
        for _ in range(100): # Reduced from 1000 for faster testing
            with tf.GradientTape() as tape:
                y = model(x)
                loss = tf.reduce_mean(y)
            
            # Backward pass
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            
    print("Test passed: Execution completed successfully within tf.name_scope.")

if __name__ == "__main__":
    test_tf_name_scope_conv()