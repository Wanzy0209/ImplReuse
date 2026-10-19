```python
import os
import tensorflow as tf

# Check for GPU availability
# PyTorch: torch.cuda.is_available()
# TensorFlow: tf.config.list_physical_devices('GPU')
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'

# Load the model
# PyTorch: torch._inductor.aoti_load_package loads a compiled model package.
# TensorFlow: tf.saved_model.load is the standard way to load a saved model.
# Note: Assuming the model is saved in TensorFlow format (e.g., SavedModel) instead of .pt2
model = tf.saved_model.load(os.path.join(os.getcwd(), "model"))

# Generate random input tensor
# PyTorch: torch.randn(8, 10, device=device)
# TensorFlow: tf.random.normal((8, 10)) placed on the specific device.
# Note: The provided context suggested tf.keras.backend.flatten, but that is for reshaping, 
# not generating random numbers. tf.random.normal is the correct equivalent to torch.randn.
with tf.device(device):
    input_tensor = tf.random.normal((8, 10))

# Run inference
output = model(input_tensor)
print(output)
```