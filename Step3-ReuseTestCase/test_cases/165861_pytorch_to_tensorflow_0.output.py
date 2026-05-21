import torch
import tensorflow as tf
import sys

def test_yiq_to_rgb_large_dims():
    """
    Adapted test case for tf.image.yiq_to_rgb based on PyTorch F.pad issue 165861.
    Verifies if the operation handles dimensions larger than uint16 max (2**16).
    """
    
    # Check for GPU availability to match the original CUDA context
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    print(f"Running on device: {device_name}")

    with tf.device(device_name):
        # --- Test Case 1: Batch dimension > 2**16 ---
        # Original PyTorch breaking case: x = torch.rand(2**16, 2, device="cuda")
        # Adaptation: yiq_to_rgb requires last dimension to be 3 (YIQ channels).
        print("\nTest 1: Batch dimension = 2**16")
        x1 = tf.random.uniform((2**16, 3), minval=0, maxval=1)
        try:
            y1 = tf.image.yiq_to_rgb(x1)
            print("yiq_to_rgb with large batch dim ok")
        except Exception as e:
            print(f"yiq_to_rgb failed: {e}")

        # --- Test Case 2: Middle dimension > 2**16 ---
        # Original PyTorch breaking case: x = torch.rand(1, 2**16, 2, device="cuda")
        print("\nTest 2: Middle dimension = 2**16")
        x2 = tf.random.uniform((1, 2**16, 3), minval=0, maxval=1)
        try:
            y2 = tf.image.yiq_to_rgb(x2)
            print("yiq_to_rgb with large middle dim ok")
        except Exception as e:
            print(f"yiq_to_rgb failed: {e}")

        # --- Test Case 3: Control (Large total elements, but dims < 2**16) ---
        # Original PyTorch fine case: x = torch.rand(2**16 - 1, 200, device="cuda")
        print("\nTest 3: Large total elements, dims < 2**16")
        x3 = tf.random.uniform((2**16 - 1, 200, 3), minval=0, maxval=1)
        try:
            y3 = tf.image.yiq_to_rgb(x3)
            print("yiq_to_rgb with large total elements ok")
        except Exception as e:
            print(f"yiq_to_rgb failed: {e}")

if __name__ == "__main__":
    test_yiq_to_rgb_large_dims()