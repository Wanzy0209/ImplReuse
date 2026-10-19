```python
import os
import tensorflow as tf

# Conversion: Check for CUDA/GPU availability
# PyTorch uses torch.cuda.is_available()
# TensorFlow uses tf.config.list_physical_devices('GPU')
gpus = tf.config.list_physical_devices('GPU')
device = "/GPU:0" if gpus else "/CPU:0"

# Conversion: Load compiled model package
# PyTorch uses torch._inductor.aoti_load_package for AOT compiled artifacts
# TensorFlow uses tf.saved_model.load for SavedModel format
# Note: Assuming 'model' is a directory containing a TensorFlow SavedModel
model = tf.saved_model.load(os.path.join(os.getcwd(), "model"))

# Conversion: Generate random input and run inference
# PyTorch: torch.randn(8, 10, device=device)
# TensorFlow: tf.random.normal((8, 10)) inside a device context
with tf.device(device):
    output = model(tf.random.normal((8, 10)))

print(output)
```