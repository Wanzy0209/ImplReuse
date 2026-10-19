```python
import tensorflow as tf
import numpy as np

# Load MobileNetV2 model without pre-trained weights
# Conversion: torchvision.models.mobilenet_v2 -> tf.keras.applications.MobileNetV2
model = tf.keras.applications.MobileNetV2(weights=None)

# Generate random input tensor
# Conversion: torch.rand -> tf.random.uniform
# Note: PyTorch uses (N, C, H, W) format, while TensorFlow uses (N, H, W, C)
x = tf.random.uniform((1, 224, 224, 3))

# Export the model to a static graph
# Conversion: torch.export.export -> tf.function
# tf.function traces the Python function to create a TensorFlow Graph, 
# analogous to capturing the graph in torch.export.
exported_model = tf.function(model)

# Assert that the outputs of the original model and the exported model are close
# Conversion: torch.testing.assert_close -> np.testing.assert_allclose
# We compare the eager execution (model) with the graph execution (exported_model)
np.testing.assert_allclose(model(x).numpy(), exported_model(x).numpy())
```