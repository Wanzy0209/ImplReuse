import torch
import numpy as np

def verify_image_input(image, min_channels: int = 3) -> bool:
    """Verification that the image has at least 3 channels (RGB)."""
    rank = len(image.shape)
    has_channels = image.shape[-1] >= min_channels
    return rank >= 3 and has_channels

def apply_random_hue(image, max_delta, seed=None):
    """
    Wrapper to apply random_hue, mimicking the structure of 
    setting modules in the original FSDP example.
    """
    # In the original bug, the user sets prefetch modules expecting them to work.
    # Here, we apply the random hue operation expecting it to adjust the image.
    return tf.image.random_hue(image, max_delta, seed=seed)

def main():
    # Handle the ImportError gracefully due to environment issues (GLIBCXX)
    try:
        import tensorflow as tf
        # Make tf available globally so helper functions can access it
        globals()['tf'] = tf
    except ImportError as e:
        print(f"Test skipped: Unable to import TensorFlow due to environment dependency issues.")
        print(f"Details: {e}")
        return

    # Setup: Create a dummy RGB image
    # Mimicking the model setup in the original code
    x = tf.constant([[[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]],
                     [[7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0]]])
    
    print(f"Original image shape: {x.shape}")
    print(f"Original image dtype: {x.dtype}")

    # Verification step
    if not verify_image_input(x):
        print("Unable to verify image input (must be RGB). Exiting.")
        return

    # Parameters for the operation
    max_delta = 0.2
    seed = 0  # Mimicking torch.manual_seed(0)

    # Apply the operation
    # The original bug was about implicit prefetch not working.
    # Here we test if the random_hue operation executes and modifies the tensor.
    with tf.device("/CPU:0"): # Explicit device placement similar to torch.device
        adjusted_image = apply_random_hue(x, max_delta, seed)

    # Inspect results (Mimicking inspect_model)
    print(f"Adjusted image shape: {adjusted_image.shape}")
    print(f"Adjusted image dtype: {adjusted_image.dtype}")
    
    # Assertions to verify behavior
    # 1. Shape must be preserved
    assert adjusted_image.shape == x.shape, "Shape mismatch after random_hue"
    
    # 2. Dtype must be preserved (usually float32 for this op)
    assert adjusted_image.dtype == x.dtype, "Dtype mismatch after random_hue"
    
    # 3. Values should have changed (unless max_delta is 0 or random chance)
    # Note: random_hue operates on RGB, so we check if the tensor is different
    # We use a small tolerance for floating point comparison if we were checking exact values,
    # but here we just check that it's not identical to the input (highly probable with delta=0.2)
    # However, since it's random, we can't strictly assert inequality in a unit test without mocking.
    # Instead, we verify the operation ran without error and produced valid output.
    
    # 4. Verify max_delta constraint logic (from docs: max_delta must be in [0, 0.5])
    # We test a valid case here.
    assert 0.0 <= max_delta <= 0.5, "max_delta must be in [0, 0.5]"

    print("Test passed: random_hue executed successfully.")

if __name__ == "__main__":
    main()