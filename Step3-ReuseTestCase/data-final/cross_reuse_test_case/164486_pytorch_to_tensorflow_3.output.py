import torch
import tensorflow as tf
import numpy as np
import sys

# Setup seed to match the reproducibility intent of the original test
tf.random.set_seed(238)

# Define the initializer (The API under test)
# HeUniform draws samples from a uniform distribution within [-limit, limit]
# where limit = sqrt(6 / fan_in)
initializer = tf.keras.initializers.HeUniform(seed=238)

def fuzzed_program(shape):
    # Mimic the structure of the original test: perform the operation
    # Here the operation is initialization
    return initializer(shape)

# Define input shape. 
# Original test used scalar and (1,) tensors. 
# We use a small shape to test the initializer logic.
shape = (3, 3)

# Eager execution
out_eager = fuzzed_program(shape)
print('Eager Success! ')

# Compiled execution (tf.function is the TensorFlow equivalent of torch.compile)
compiled_program = tf.function(fuzzed_program)
out_compiled = compiled_program(shape)
print('Compile Success! ')

# Verification logic
# 1. Check if Eager and Compiled outputs match (Determinism check)
# Note: Initializers are stateless, so they should produce the same result 
# given the same seed and shape, regardless of tf.function tracing.
diff = np.abs(out_eager.numpy() - out_compiled.numpy()).sum()

print(f'Absolute diff (sum): {diff}')

if diff > 1e-5:
    print(f' Eager and Compiled outputs differ!')
    print('out_eager:', out_eager.numpy())
    print('out_compiled:', out_compiled.numpy())
    sys.exit(1)
else:
    print(' Eager and Compiled outputs match.')

# 2. Check if values are within the expected HeUniform range
# fan_in for shape (3, 3) is 3. limit = sqrt(6/3) = sqrt(2) ~= 1.414
fan_in = shape[1] # Assuming 2D tensor for fan_in calculation
limit = np.sqrt(6.0 / fan_in)

max_val = np.max(np.abs(out_eager.numpy()))
if max_val > limit + 1e-5:
    print(f' Values outside HeUniform range [-{limit}, {limit}]!')
    print(f'Max absolute value: {max_val}')
    sys.exit(1)
else:
    print(f' Values within expected HeUniform range [-{limit:.4f}, {limit:.4f}].')