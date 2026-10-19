```python
import os
import tensorflow as tf

# Check for GPU availability
# TensorFlow uses device strings like '/GPU:0' or '/CPU:0'
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Load SavedModel (equivalent to loading a compiled package)
# Note: TensorFlow SavedModels are directories, not single files like .pt2
model = tf.saved_model.load(os.path.join(os.getcwd(), "model"))

# Run inference on the specified device
with tf.device(device):
    # Generate random input (equivalent to torch.randn)
    # Note: tf.random.normal does not take a device argument, placement is handled by context
    output = model(tf.random.normal((8, 10)))

print(output)
```