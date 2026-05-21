import torch
import tensorflow as tf

class TestDeviceMaskNaN(tf.test.TestCase):
    def test_4d_mask_on_gpu(self):
        """
        Test case adapted from PyTorch issue #167515.
        Leverages tf.test.is_built_with_cuda to handle device availability,
        mirroring the device-specific nature of the original bug.
        """
        # Leverage the similar API to check for device availability
        # This replaces the parametrize logic in the original PyTorch test
        if not tf.test.is_built_with_cuda():
            self.skipTest("Test is only applicable on GPU")

        # Replicate the 4D attention mask logic from the bug report
        attention_mask = tf.constant([
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]]],
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [ True,  True,  True, False],
              [ True,  True,  True,  True]]]], dtype=tf.bool)

        # Run on GPU
        # In TensorFlow, standard execution is equivalent to torch.no_grad()
        # (gradients are only tracked inside tf.GradientTape)
        with tf.device("/GPU:0"):
            # Simulate a simple operation that might be affected by the mask
            # (e.g. masked multiplication)
            logits = tf.random.normal([2, 1, 4, 4])
            mask_float = tf.cast(attention_mask, logits.dtype)
            output = logits * mask_float

            # Assert no NaNs
            self.assertFalse(tf.reduce_any(tf.math.is_nan(output)), "Logits contain NaN values")

if __name__ == "__main__":
    tf.test.main()