```python
import tensorflow as tf

worker_name = "trace"

# Conversion: torch.profiler.profile with on_trace_ready maps to 
# tf.profiler.experimental.start. The log directory is passed directly.
# Note: TensorFlow handles CPU and GPU activities automatically based on availability.
tf.profiler.experimental.start(".")

# Conversion: Explicit start/stop methods map to experimental start/stop functions.
# The 'profile' object state is managed globally in TF's experimental profiler.

# Conversion: device="cuda" maps to tf.device('/GPU:0')
with tf.device('/GPU:0'):
    # Conversion: torch.randn maps to tf.random.normal
    x = tf.random.normal((2,))
    y = x + 21
    z = x * 15

tf.profiler.experimental.stop()
```