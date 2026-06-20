import torch
import sys

# Handle the environment dependency issue (GLIBC version mismatch)
try:
    import tensorflow as tf
    print("TensorFlow Version:", tf.__version__)
except ImportError as e:
    print(f"Skipping test: Failed to import TensorFlow due to environment incompatibility.")
    print(f"Error: {e}")
    sys.exit(0)

# Replicate the tensor creation logic from the original bug report
# Original: tensor1 = torch.randint(..., size=(9, 3, 7), dtype=torch.int16)
# Adaptation: Create a tensor with the same shape and dtype (int16), which is invalid for TF's inputs (expects float32)
tensor1 = tf.random.uniform((9, 3, 7), minval=-100, maxval=100, dtype=tf.int16)

# Original: tensor2 = torch.randint(..., size=(1, 6, 4, 8), dtype=torch.bool)
# Adaptation: Create a tensor with the same shape and dtype (bool), which is invalid for TF's sequence_length (expects int32)
tensor2 = tf.random.uniform((1, 6, 4, 8), minval=0, maxval=2, dtype=tf.bool)

# Original: input[1] contained a huge integer 154691921484029491302139942063978250367
# Adaptation: Use this huge integer for the beam_width parameter
huge_int = 154691921484029491302139942063978250367

try:
    # Call the similar API (tf.compat.v1.nn.ctc_beam_search_decoder)
    # We pass the malformed tensors and the huge integer to test robustness
    decoded, log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(
        inputs=tensor1,
        sequence_length=tensor2,
        beam_width=huge_int
    )
    
    # If the API doesn't crash, print the result
    print("API call succeeded. Result type:", type(decoded))
    print("Decoded:", decoded)

except Exception as e:
    # Catch and report exceptions to verify behavior (crash vs. error handling)
    print(f"Exception caught: {type(e).__name__}")
    print(f"Error details: {e}")