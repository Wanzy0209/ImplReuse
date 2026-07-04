```python
import os
import tensorflow as tf

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Conversion: torch._inductor.aoti_load_package loads a compiled PyTorch model.
# In TensorFlow, the equivalent is loading a SavedModel directory.
# Note: The file extension/format changes from .pt2 to a SavedModel directory.
model = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model"))

# Conversion: torch.randn generates random numbers.
# tf.random.normal is the equivalent. We use tf.device context for placement.
with tf.device(device):
    output = model(tf.random.normal((8, 10)))

print(output)
```