import torch
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Skipping test due to environment incompatibility (e.g., GLIBC version).")
    import sys
    sys.exit(0)

print(tf.__version__)

# Replicate the tensor creation logic from the PyTorch bug report
# PyTorch: tensor1 = torch.randint(low=-100, high=100, size=(9, 3, 7), dtype=torch.int16)
tensor1 = tf.random.uniform(
    minval=-100,
    maxval=100,
    shape=(9, 3, 7),
    dtype=tf.int16
)

# PyTorch: tensor2 = torch.randint(low=0, high=2, size=(1, 6, 4, 8), dtype=torch.bool)
# Note: TF random uniform doesn't support bool directly, so we generate int and cast
tensor2 = tf.cast(
    tf.random.uniform(
        minval=0,
        maxval=2,
        shape=(1, 6, 4, 8),
        dtype=tf.int32
    ),
    dtype=tf.bool
)

# Extract the anomalous arguments from the original bug report
# Original: input = [[[], 154691921484029491302139942063978250367, ()],{},[tensor1,tensor2],{}]
# The huge integer is mapped to beam_width in the similar API
huge_int = 154691921484029491302139942063978250367

# Adapt the test case to tf.nn.ctc_beam_search_decoder
# The original bug passed invalid types/shapes and a huge integer to trigger a segfault.
# We pass these to the TensorFlow equivalent to verify its behavior.
try:
    # inputs expects float32/64, we pass int16 (mimicking tensor1)
    # sequence_length expects int32 1D, we pass bool 4D (mimicking tensor2)
    # beam_width expects a reasonable int, we pass the huge_int
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        inputs=tensor1,
        sequence_length=tensor2,
        beam_width=huge_int
    )
    print("Test Case Result: Success (No Crash)")
    print("Decoded:", decoded)
except Exception as e:
    # TensorFlow is expected to raise an error for invalid inputs rather than segfaulting,
    # but we capture it to verify the behavior.
    print(f"Test Case Result: Caught Exception - {type(e).__name__}")
    print(f"Message: {e}")