import torch
import tensorflow as tf
import numpy as np

def test_large_tensor_replacement():
    # 1. Replace torch.randint with torch.amax
    # Original: mask = torch.randint(0, 20, (4, 87, 1056, 736), device="cuda")
    # We create a large random tensor and apply amax to generate the 'mask' variable.
    # To get the shape (4, 87, 1056, 736), we reduce a tensor of shape (1, 4, 87, 1056, 736) along dim 0.
    input_data = torch.randn(1, 4, 87, 1056, 736, device="cuda")
    mask = torch.amax(input_data, dim=0)

    # 2. Replace torch.tensor with torch.isnan
    # Original: to_apply = torch.tensor([True, False, False, True], device="cuda")
    # We create a tensor with NaNs to generate the boolean mask [True, False, False, True].
    temp_tensor = torch.tensor([float('nan'), 1.0, 2.0, float('nan')], device="cuda")
    to_apply = torch.isnan(temp_tensor)

    # 3. Replace Tensor indexing (mask[to_apply]) with tf.compat.v1.distributions.Categorical
    # The original operation indexed the large tensor. We replace this with the similar API.
    # We convert the PyTorch tensor to a TensorFlow tensor to use with the TF API.
    # We attempt to place it on GPU to match the original bug context.
    with tf.device("/GPU:0"):
        # Convert PyTorch tensor to NumPy, then to TensorFlow tensor
        mask_tf = tf.constant(mask.cpu().numpy())
        
        # Instantiate the Categorical distribution using the large tensor as logits.
        # This replaces the indexing operation mask[to_apply].
        dist = tf.compat.v1.distributions.Categorical(logits=mask_tf)
        
        # Verify the distribution was created with the correct shape
        # The original mask shape is (4, 87, 1056, 736)
        # Categorical interprets the last dimension as the event dimension.
        # Batch shape should be (4, 87, 1056)
        assert dist.batch_shape == (4, 87, 1056)
        assert dist.event_shape == (736,)

if __name__ == "__main__":
    test_large_tensor_replacement()