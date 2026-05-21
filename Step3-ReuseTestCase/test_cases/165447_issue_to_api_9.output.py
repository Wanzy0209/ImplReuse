import torch
import tensorflow as tf

# Adapted from PyTorch Issue 165447: AOT Precompile serialization failed when running multiple times.
# The original issue tests the lifecycle of a compiled function: creation, saving, resetting, loading, and execution.
# This test adapts that logic to tf.summary.create_noop_writer, verifying that the writer
# can be instantiated, used as a context manager, and closed repeatedly without error.

# Setup dummy data (analogous to sample_inputs in the original issue)
dummy_data = tf.constant(1.0)

# Run the lifecycle test multiple times to address the "running multiple times" aspect of the bug
for i in range(2):
    # 1. Create the writer (analogous to aot_compile)
    writer = tf.summary.create_noop_writer()
    assert writer is not None

    # 2. Use the writer in a context manager (analogous to running the compiled function)
    # The original bug report relies heavily on context managers (with open, with set_stance).
    # The similar API (create_noop_writer) provides a context manager via as_default().
    with writer.as_default():
        # Perform a write operation (analogous to forward pass)
        # Since it is a noop writer, this performs no I/O, but we verify the API accepts the call.
        tf.summary.scalar("dummy_metric", dummy_data, step=i)

    # 3. Cleanup/Reset (analogous to torch._dynamo.reset)
    writer.close()

# Final assertion to ensure stability after multiple runs
final_writer = tf.summary.create_noop_writer()
assert final_writer is not None
final_writer.close()

print("Test passed: Noop writer handles multiple lifecycle runs successfully.")