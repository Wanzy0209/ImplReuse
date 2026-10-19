```python
import os
import tensorflow as tf

# Conversion: Check for GPU availability (TF equivalent of torch.cuda.is_available)
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Conversion: Load a TensorFlow SavedModel instead of a TorchInductor AOTI package
# Note: TensorFlow uses the SavedModel directory format, not .pt2 files
model = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model"))

# Conversion: Use tf.random.normal instead of torch.randn
# Conversion: Use tf.device context manager to place operations on the selected device
with tf.device(device):
    output = model(tf.random.normal((8, 10)))

print(output)
```