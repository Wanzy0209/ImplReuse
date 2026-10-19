```python
import os
import tensorflow as tf

# Conversion: torch.cuda.is_available() -> tf.config.list_physical_devices('GPU')
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Conversion: torch._inductor.aoti_load_package -> tf.saved_model.load
# Note: Loading a standard TensorFlow SavedModel instead of a PyTorch .pt2 package
model = tf.saved_model.load(os.path.join(os.getcwd(), "model"))

# Conversion: torch.randn -> tf.random.normal
# Note: Explicitly placing the operation on the device
with tf.device(device):
    output = model(tf.random.normal((8, 10)))

print(output)
```