```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, WeightNormalization
from tensorflow.keras.optimizers import SGD

if __name__ == "__main__":
    d = 65
    
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: PyTorch uses NCHW by default, TF uses NHWC. 
    # We use data_format='channels_first' to preserve the input shape (1, 2, 32, 32).
    # Conversion: .cuda() -> tf.device context
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device):
        x = tf.random.normal((1, 2, 32, 32))
        
        # Conversion: nn.Conv2d -> Conv2D
        # Conversion: weight_norm -> WeightNormalization wrapper
        layer = Conv2D(filters=d, kernel_size=2, data_format='channels_first')
        model = WeightNormalization(layer)
        
        # Conversion: torch.compile -> tf.function for graph optimization
        compiled_model = tf.function(model)
        
        # Conversion: SGD(model.parameters()) -> SGD()
        opt = SGD()
        
        for _ in range(1000):
            # Conversion: PyTorch autograd -> tf.GradientTape
            with tf.GradientTape() as tape:
                output = compiled_model(x)
                loss = tf.reduce_mean(output)
            
            # Conversion: backward() -> tape.gradient()
            grads = tape.gradient(loss, model.trainable_variables)
            
            # Conversion: opt.step() -> opt.apply_gradients()
            opt.apply_gradients(zip(grads, model.trainable_variables))
            
            # Conversion: opt.zero_grad() -> Not needed in TF (GradientTape is fresh every iteration)
```