```python
import os
import tensorflow as tf

# Conversion: Check for GPU availability in TensorFlow
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Conversion: Load a TensorFlow SavedModel (equivalent to loading a compiled package)
# Note: Assuming 'model.pt2' is replaced by a standard TensorFlow SavedModel directory
model = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model"))

with tf.device(device):
    # Conversion: Generate random tensor with standard normal distribution
    # Note: In TensorFlow, device placement is handled by the context manager
    output = model(tf.random.normal((8, 10)))

print(output)
```