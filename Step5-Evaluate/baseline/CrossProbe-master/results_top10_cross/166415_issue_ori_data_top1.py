```python
import os
import tensorflow as tf

# Conversion: torch.cuda.is_available() -> tf.config.list_physical_devices('GPU')
# Note: TensorFlow device strings typically use "/GPU:0" or "/CPU:0"
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Conversion: torch._inductor.aoti_load_package -> tf.saved_model.load
# Note: PyTorch uses .pt2 packages, TensorFlow typically loads SavedModel directories
model = tf.saved_model.load(os.path.join(os.getcwd(), "model"))

# Conversion: torch.randn -> tf.random.normal
# Note: Device placement in TF is often handled via context managers
with tf.device(device):
    input_tensor = tf.random.normal((8, 10))
    output = model(input_tensor)

print(output)
```