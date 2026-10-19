```python
import os
# Conversion: PyTorch specific CUDA allocator configuration.
# TensorFlow uses the BFC allocator and does not support this specific environment variable.
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "roundup_power2_divisions:[>:1]"

import tensorflow as tf

MB = 1024 * 1024

# Conversion: torch.empty -> tf.empty
# Conversion: device='cuda' -> tf.device('/GPU:0')
# Note: tf.empty creates a tensor with uninitialized entries, similar to torch.empty
with tf.device('/GPU:0'):
    t = tf.empty([514 * MB], dtype=tf.int8)

# Conversion: torch.cuda.memory_summary -> tf.config.experimental.get_memory_info
# Note: TensorFlow does not provide a formatted string summary, so we print the memory info dictionary.
if tf.config.list_physical_devices('GPU'):
    mem_info = tf.config.experimental.get_memory_info('GPU:0')
    print(f"Current Memory Usage: {mem_info['current'] / (1024**2):.2f} MB")
    print(f"Peak Memory Usage: {mem_info['peak'] / (1024**2):.2f} MB")
else:
    print("No GPU available.")
```