import torch
import tensorflow as tf
import numpy as np

# Helper to create tf.distribute.DistributedValues (Replacement for torch.randn)
def create_distributed_values(shape):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Using a variable to easily get a DistributedValues object
        v = tf.Variable(np.random.randn(*shape).astype(np.float32))
        # strategy.run returns a DistributedValues
        def get_val():
            return v.read_value()
        return strategy.run(get_val)

# Dimensions
m, k, n = 20120, 1536, 512

# Create data using the replacement API
# We need to convert these to torch tensors for the subsequent torch operations
# This bridges the gap between the replacement API and the rest of the logic.
def to_torch(dist_val):
    # Extract values from the first replica
    return torch.from_numpy(dist_val.experimental_local_results()[0].numpy())

a = to_torch(create_distributed_values((m, n)))
mat1 = to_torch(create_distributed_values((m, k)))
mat2 = to_torch(create_distributed_values((k, n)))

# Replacement: torch.addmm -> torch.fmin
# Original: f = lambda a, mat1, mat2: torch.addmm(a, mat1, mat2)
# New: fmin takes 2 inputs. We use a and mat1.
f = lambda a, mat1, mat2: torch.fmin(a, mat1)

# Replacement: torch._inductor.config -> tf.compat.v1.train
# Original: with inductor_config.patch(...)
# We use tf.compat.v1.train.AdamOptimizer to represent the configuration context
optimizer = tf.compat.v1.train.AdamOptimizer()

# Similar API: tf.keras.callbacks.TerminateOnNaN
callback = tf.keras.callbacks.TerminateOnNaN()

# Execution
# Run the function
result = f(a, mat1, mat2)

# Check result with the callback
# The callback expects a loss. We'll use the sum of the result as a proxy loss.
loss = torch.sum(result).item()

# Simulate batch end
callback.on_batch_end(0, {'loss': loss})

# Assertion
if not np.isnan(loss) and not np.isinf(loss):
    print("Test Passed: No NaN/Inf detected.")
else:
    print("Test Failed: NaN/Inf detected.")