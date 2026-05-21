import torch
import tensorflow as tf

# Check for GPU availability to match the context of the original bug (CUDA)
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'
print(f"Running on device: {device}")

with tf.device(device):
    # Test Case 1: Batch dimension > 2**16 (65536)
    # PyTorch equivalent: x = torch.rand(2**16, 2, device="cuda")
    # This case triggered the CUDA error in PyTorch reflect padding.
    try:
        indices = tf.random.uniform((2**16, 2), minval=0, maxval=10, dtype=tf.int32)
        out = tf.keras.backend.one_hot(indices, num_classes=10)
        print("Test 1 (Batch dim > 65536): OK")
    except Exception as e:
        print(f"Test 1 (Batch dim > 65536): FAILED - {e}")

    # Test Case 2: Middle dimension > 2**16
    # PyTorch equivalent: x = torch.rand(1, 2**16, 2, device="cuda")
    try:
        indices = tf.random.uniform((1, 2**16, 2), minval=0, maxval=10, dtype=tf.int32)
        out = tf.keras.backend.one_hot(indices, num_classes=10)
        print("Test 2 (Middle dim > 65536): OK")
    except Exception as e:
        print(f"Test 2 (Middle dim > 65536): FAILED - {e}")

    # Test Case 3: Large dimension but within limit (Control)
    # PyTorch equivalent: x = torch.rand(2**16 - 1, 200, device="cuda")
    # This case worked fine in PyTorch.
    try:
        indices = tf.random.uniform((2**16 - 1, 200), minval=0, maxval=10, dtype=tf.int32)
        out = tf.keras.backend.one_hot(indices, num_classes=10)
        print("Test 3 (Dim < 65536, large total): OK")
    except Exception as e:
        print(f"Test 3 (Dim < 65536, large total): FAILED - {e}")