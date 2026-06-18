import torch
import os
import tensorflow as tf
from tensorflow.experimental import dtensor

# Setup environment variable for the similar API
# Using a valid sorted BNS list to ensure success, mimicking the setup in the original bug
os.environ['DTENSOR_JOBS'] = "/bns/worker0,/bns/worker1,/bns/worker2"

def fuzzed_program():
    """
    Mimics the structure of the original fuzzed_program.
    Instead of tensor operations, it calls the similar API (dtensor.jobs).
    """
    return dtensor.jobs()

# 1. Eager execution
result_eager = fuzzed_program()
print(f' eager success: {result_eager}')

# 2. Compile execution
# In TensorFlow, torch.compile is analogous to tf.function (graph mode)
# We use fullgraph-like behavior by default in tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program()
print(f' compile success: {result_compiled}')

# 3. Check for divergence
# The original bug reported a divergence between eager and compile modes.
# We assert that the results match to ensure the similar API behaves consistently.
assert result_eager == result_compiled, "Divergence detected between eager and compiled modes"

# Clean up
if 'DTENSOR_JOBS' in os.environ:
    del os.environ['DTENSOR_JOBS']