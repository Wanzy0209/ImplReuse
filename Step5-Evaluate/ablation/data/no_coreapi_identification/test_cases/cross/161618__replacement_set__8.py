import torch
import tensorflow as tf
import triton
import numpy as np

# Leverage Similar API: tf.keras.callbacks.TerminateOnNaN
# This callback is used to detect and handle the failure state (NaN/Inf) 
# analogous to the Inductor failure described in the bug report.
nan_callback = tf.keras.callbacks.TerminateOnNaN()

m = 20120
k = 1536
n = 512

# Replacement: torch.randn -> tf.tpu.experimental.embedding.Adam
# Replacing the random tensor generation with the Adam optimizer class.
# Note: This is a structural replacement as per instructions.
a = tf.tpu.experimental.embedding.Adam(learning_rate=0.001)
mat1 = tf.tpu.experimental.embedding.Adam(learning_rate=0.001)
mat2 = tf.tpu.experimental.embedding.Adam(learning_rate=0.001)

# Replacement: torch.addmm -> torch.amax
# Replacing the matrix addition/multiplication operation with a max reduction.
# We use a dummy tensor for the operation since inputs are now Optimizers.
f = lambda a, mat1, mat2: torch.amax(torch.randn(1))

# Replacement: torch._inductor.config -> tf.distribute.DistributedValues
# We use a distribution strategy scope to provide the context for DistributedValues,
# replacing the configuration patching context.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Instantiate the replacement class to represent the distributed configuration state
    config_replacement = tf.distribute.DistributedValues(
        [tf.Variable(0.0)], strategy
    )

    # Original logic: Compile and run
    try:
        # torch.compile is retained as it was not in the replacement list
        compiled = torch.compile(f, dynamic=False)

        # Execute the compiled function
        # Passing None/dummy values as inputs are Optimizers, not Tensors
        result = compiled(None, None, None)

        # Check for NaNs to leverage the Similar API (TerminateOnNaN)
        if torch.isnan(result).any():
            nan_callback.on_batch_end(0, {'loss': float('nan')})

    except Exception as e:
        # Bug reproduction: Inductor fails.
        # We use the callback to signal this failure state.
        print(f"Execution failed: {e}")
        # Trigger the callback logic with a NaN loss
        nan_callback.on_batch_end(0, {'loss': float('nan')})