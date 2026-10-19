```python
import tensorflow as tf
from tensorflow.keras import applications

# Load MobileNetV2 without pre-trained weights
# PyTorch: torchvision.models.mobilenet_v2(weights=None)
model = applications.MobileNetV2(weights=None)

# Create a random input tensor
# PyTorch uses (Batch, Channels, Height, Width) format
# TensorFlow uses (Batch, Height, Width, Channels) format
x = tf.random.uniform((1, 224, 224, 3))

# Export the model to a static graph representation
# PyTorch: torch.export.export captures the computation graph
# TensorFlow: tf.function converts the model to a static graph
exported_model = tf.function(model)

# Assert that the original model and the exported model produce the same output
# PyTorch: torch.testing.assert_close
# TensorFlow: tf.debugging.assert_near
tf.debugging.assert_near(model(x), exported_model(x))
```