```python
import tensorflow as tf

# Conversion: torch.cuda.is_available() checks for GPU presence
# In TensorFlow, we check if the list of physical GPUs is not empty
print(f"CUDA Available: {len(tf.config.list_physical_devices('GPU')) > 0}")  # Returns False

# Conversion: torch.cuda.device_count() returns the number of GPUs
# In TensorFlow, we count the length of the list of physical GPUs
print(f"GPU Count: {len(tf.config.list_physical_devices('GPU'))}")       # Returns 0
```