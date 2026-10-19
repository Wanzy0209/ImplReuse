```python
import tensorflow as tf

# Conversion: Implementing Weight Normalization manually as it is not a standard layer in core Keras
# PyTorch: torch.nn.utils.parametrizations.weight_norm
class Conv2DWithWeightNorm(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        # Conversion: PyTorch Conv2d has bias=True by default
        self.use_bias = True

    def build(self, input_shape):
        # Conversion: PyTorch Conv2d weight shape is (out_channels, in_channels, kH, kW)
        # TensorFlow Conv2D kernel shape is (kH, kW, in_channels, out_channels)
        kernel_shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters)

        # Initialize v (direction vector)
        # Conversion: PyTorch initializes v with the original weight values
        initial_v = tf.keras.initializers.GlorotUniform()(shape=kernel_shape)
        self.v = self.add_weight(name='v', shape=kernel_shape, initializer=tf.constant_initializer(initial_v), trainable=True)

        # Initialize g (magnitude scalar)
        # Conversion: PyTorch initializes g with the norm of the original weight
        # Norm is computed over dimensions corresponding to input channels and kernel size (dim=0 in PyTorch)
        initial_norm = tf.linalg.norm(initial_v, axis=[0, 1, 2])
        self.g = self.add_weight(name='g', shape=(self.filters,), initializer=tf.constant_initializer(initial_norm), trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros')
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs):
        # Reconstruct weight: w = g * (v / ||v||)
        # Conversion: PyTorch dim=0 corresponds to normalizing over input channels and spatial dims for each output channel
        norm = tf.linalg.norm(self.v, axis=[0, 1, 2], keepdims=True)
        # Add epsilon for numerical stability
        w = self.g * (self.v / (norm + 1e-12))

        # Conversion: PyTorch Conv2d default padding is 0 (VALID in TF)
        # Conversion: PyTorch Conv2d default stride is 1
        # Conversion: PyTorch uses NCHW format, TensorFlow uses NHWC
        outputs = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding='VALID')

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

if __name__ == "__main__":
    d = 65
    
    # Conversion: torch.randn((1, 2, 32, 32)).cuda() -> tf.random.normal with device placement
    # Conversion: PyTorch (N, C, H, W) -> TensorFlow (N, H, W, C)
    with tf.device('/GPU:0'):
        x = tf.random.normal((1, 32, 32, 2))
        
        # Conversion: weight_norm(nn.Conv2d(2, d, 2)) -> Custom Layer
        # Conversion: .train() -> Keras layers are in training mode by default
        # Conversion: torch.compile -> tf.function
        layer = Conv2DWithWeightNorm(filters=d, kernel_size=2)
        model = tf.function(layer)
        
        # Conversion: SGD(model.parameters()) -> tf.keras.optimizers.SGD
        opt = tf.keras.optimizers.SGD()
        
        for _ in range(1000):
            # Conversion: PyTorch autograd -> TensorFlow GradientTape
            with tf.GradientTape() as tape:
                # Conversion: model(x).mean() -> tf.reduce_mean(model(x))
                loss = tf.reduce_mean(model(x))
            
            # Conversion: backward() -> tape.gradient
            grads = tape.gradient(loss, layer.trainable_variables)
            
            # Conversion: opt.step() -> opt.apply_gradients
            opt.apply_gradients(zip(grads, layer.trainable_variables))
            
            # Conversion: opt.zero_grad() -> Handled automatically by GradientTape
```