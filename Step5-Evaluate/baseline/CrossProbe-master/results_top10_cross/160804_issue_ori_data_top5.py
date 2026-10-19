```python
import tensorflow as tf

# Conversion: torch.cuda.is_available() -> len(tf.config.list_physical_devices('GPU')) > 0
print(f"CUDA Available: {len(tf.config.list_physical_devices('GPU')) > 0}")  # Returns False

# Conversion: torch.cuda.device_count() -> len(tf.config.list_physical_devices('GPU'))
print(f"GPU Count: {len(tf.config.list_physical_devices('GPU'))}")           # Returns 0
```