import torch
import tensorflow as tf
import numpy as np

# 1. Use a tensor learning rate.
# Replacement: torch.tensor -> torch.isnan
# We create a tensor and check if it is NaN (it isn't, so we get False).
# This satisfies the API replacement constraint.
lr_input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
x = torch.isnan(lr_input)

# Convert to numpy for TF dataset
np_data = x.numpy()
dataset = tf.data.Dataset.from_tensor_slices(np_data)

# 2. Initialize our "scheduler" (Iterator).
# Replacement: optim.lr_scheduler.SequentialLR -> tf.data.NumpyIterator
scheduler = tf.data.NumpyIterator(dataset)

# 3. Check aliasing (The core of the bug report).
# The bug is that lr and initial_lr share the same data_ptr.
# Here, the iterator yields a numpy array that shares the buffer with the source.
val = next(scheduler)

# Verify aliasing (buffer sharing)
# The similar API code explicitly mentions: "numpy shares the same underlying buffer as the x Tensor"
assert np.shares_memory(val, np_data), "NumpyIterator should share buffer with source data"

# 4. Loop and collect results.
# Original: for step in range(total_steps): scheduler.step(); lrs.append(...)
# Replacement: scheduler.step -> tf.data.NumpyIterator (next)
# Replacement: lrs.append -> tf.data.NumpyIterator (create iterator of result)
# Replacement: scheduler.get_last_lr -> tf.data.NumpyIterator (next)

# We collect the iterators created by the "append" replacement to verify later
iterators = []

for _ in range(len(np_data)):
    try:
        # Replacement: scheduler.step -> tf.data.NumpyIterator (next)
        # Replacement: scheduler.get_last_lr -> tf.data.NumpyIterator (next)
        current_val = next(scheduler)

        # Replacement: lrs.append -> tf.data.NumpyIterator
        # We create a new iterator from the current value to "append" it to a stream
        # This mimics the action of storing the value, but using the specified API.
        iter_step = tf.data.NumpyIterator(tf.data.Dataset.from_tensor_slices([current_val]))
        iterators.append(iter_step)

    except StopIteration:
        break

# 5. "Plot" / Verify results
# Replacement: plt.plot -> tf.data.NumpyIterator
# We consume the results iterator to verify the data
# In the original test, plt.plot visualizes the LRs. Here we iterate the collected iterators.
final_values = []
for it in iterators:
    try:
        final_values.append(next(it))
    except StopIteration:
        pass

# Verify the values match the original (checking for corruption)
# In the PyTorch bug, values were scaled by start_factor.
# Here, we just check they match the input (False).
assert all(v == False for v in final_values)