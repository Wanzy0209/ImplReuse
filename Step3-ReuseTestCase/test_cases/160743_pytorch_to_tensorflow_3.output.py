import tensorflow as tf

# Set seed for reproducibility
tf.random.set_seed(0)

# Adapt input: PyTorch (C, H, W) -> TF (Batch, H, W, C)
# Original PyTorch input shape: (4, 6, 7)
# TensorFlow input shape (NHWC): (1, 6, 7, 4)
x = tf.random.normal((1, 6, 7, 4))

# Adapt parameters:
# PyTorch AvgPool2d: kernel_size=[1, 6], stride=[4, 9], ceil_mode=True
# Input H=6, W=7.
# Output H calculation: ceil((6 - 1) / 4) + 1 = 2
# Output W calculation: ceil((7 - 6) / 9) + 1 = 1
# Target Output Shape: (1, 2, 1, 4)
# To achieve this with fractional_max_pool, we set pooling_ratio to approximate the reduction.
# H reduction: 6 -> 2 (ratio 3.0)
# W reduction: 7 -> 1 (ratio 7.0)
pooling_ratio = [1.0, 3.0, 7.0, 1.0]

# Run the operation
# deterministic=True ensures the pooling regions are chosen deterministically based on seed
out, _, _ = tf.compat.v1.nn.fractional_max_pool(
    value=x,
    pooling_ratio=pooling_ratio,
    pseudo_random=False,
    overlapping=False,
    deterministic=True,
    seed=0
)

# Verify behavior: Check output shape matches the expected dimensions derived from PyTorch logic
expected_shape = (1, 2, 1, 4)
if out.shape != expected_shape:
    print(f"Shape mismatch! Expected {expected_shape}, got {out.shape}")
else:
    print("Output shape matches.")
    print(out)