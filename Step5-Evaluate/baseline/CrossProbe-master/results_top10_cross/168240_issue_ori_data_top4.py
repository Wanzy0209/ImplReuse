```python
import tensorflow as tf
import numpy as np

# Load MobileNetV2 without pre-trained weights
# Conversion: torchvision.models.mobilenet_v2 -> tf.keras.applications.MobileNetV2
model = tf.keras.applications.MobileNetV2(weights=None)

# Create random input tensor
# Conversion: PyTorch uses NCHW format (1, 3, 224, 224), TensorFlow uses NHWC (1, 224, 224, 3)
x = tf.random.uniform((1, 224, 224, 3))

# Export the model to a graph representation
# Conversion: torch.export.export creates a static graph. In TF, tf.function traces the model into a graph.
exported_model = tf.function(model)

# Assert that the original model and the exported model produce the same output
# Conversion: torch.testing.assert_close compares tensors. We use numpy's allclose for script-level assertion.
np.testing.assert_allclose(model(x).numpy(), exported_model(x).numpy())
```