import torch
import tensorflow as tf

def f(x, y):
    # Translating PyTorch operations to TensorFlow
    # y2 = torch.cat([x[:, 1:], y[:, None] + 32 * 2048], dim=1)
    y2 = tf.concat(
        [
            x[:, 1:],
            tf.expand_dims(y, 1) + 32 * 2048,
        ],
        axis=1,
    )

    # x2 = x[:, 1:, None]
    x2 = tf.expand_dims(x[:, 1:], -1)
    
    # y3 = y2[:, -1:, None]
    y3 = tf.expand_dims(y2[:, -1:], -1)

    # return (torch.cat([x2, y3], dim=1) + torch.arange(-2048, 0, device=device)[None, None, :]).reshape(1, 32 * 2048)
    return tf.reshape(
        tf.concat([x2, y3], axis=1) + tf.range(-2048, 0)[tf.newaxis, tf.newaxis, :],
        (1, 32 * 2048)
    )

# Define inputs
# PyTorch: torch.zeros(1, 32, dtype=torch.int64, device=device)
x = tf.zeros((1, 32), dtype=tf.int64)
# PyTorch: torch.zeros(1, dtype=torch.int32, device=device)
y = tf.zeros((1,), dtype=tf.int32)

# This succeeds (Baseline execution)
print("Running baseline execution...")
result_baseline = f(x, y)
print("Baseline result shape:", result_baseline.shape)

# This runs inside the similar API: tf.keras.backend.name_scope
# The original bug involved torch.compile crashing on this logic.
# Here we verify that the logic executes correctly within the name_scope context.
print("Running inside tf.keras.backend.name_scope...")
with tf.keras.backend.name_scope("bug_reproduction_scope"):
    result_scoped = f(x, y)

print("Scoped result shape:", result_scoped.shape)

# Verify that the behavior is consistent
assert tf.reduce_all(result_baseline == result_scoped).numpy()
print("Test passed: Logic executed successfully inside name_scope.")