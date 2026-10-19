```python
import tensorflow as tf

worker_name = "trace"

# Conversion: torch.profiler.profile with on_trace_ready -> tf.profiler.experimental.start
# The tensorboard_trace_handler logic is handled by the logdir argument.
# TensorFlow automatically profiles CPU and GPU activities based on where ops are executed.
tf.profiler.experimental.start(logdir=".")

# Conversion: torch.randn(2, device="cuda") -> tf.random.normal
# Note: In TensorFlow, device placement is often automatic. 
# To strictly match device="cuda", one could use 'with tf.device("/GPU:0"):' context.
x = tf.random.normal((2,))

y = x + 21
z = x * 15

# Conversion: profile.stop() -> tf.profiler.experimental.stop()
tf.profiler.experimental.stop()
```