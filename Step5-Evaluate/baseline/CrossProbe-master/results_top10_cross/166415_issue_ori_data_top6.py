```python
import os
import tensorflow as tf

# Conversion: Check for GPU availability using tf.config
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Conversion: Load a TensorFlow SavedModel instead of a Torch-TensorRT package
# Note: tf.saved_model.load expects a directory path containing the SavedModel assets
model = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model"))

# Conversion: Use tf.device context for placement and tf.random.normal for input generation
with tf.device(device):
    output = model(tf.random.normal((8, 10)))

print(output)
```