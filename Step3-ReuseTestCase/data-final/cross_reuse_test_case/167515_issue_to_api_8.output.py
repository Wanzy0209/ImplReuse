import torch
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

class TestROCMModelInference(tf.test.TestCase):
    def test_4d_attention_mask_no_nan(self):
        # Leverage the similar API to check for ROCm support.
        # This mirrors the hardware-specific testing logic of the original issue (MPS).
        if not tf.test.is_built_with_rocm():
            self.skipTest("Test is only applicable on ROCm (GPU)")

        # Use a standard model available in TensorFlow.
        # 'gpt2' is used as a proxy for 'sbintuitions/tiny-lm' to ensure runnability.
        model = TFAutoModelForCausalLM.from_pretrained("gpt2")

        # Define inputs
        input_ids = tf.constant([[0, 1, 0, 0], [0, 1, 2, 3]], dtype=tf.int32)

        # Define 4D attention mask
        attention_mask = tf.constant([
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]]],
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [ True,  True,  True, False],
              [ True,  True,  True,  True]]]], dtype=tf.bool)

        # Run inference.
        # In TensorFlow 2.x, eager execution is default and does not compute gradients,
        # effectively behaving like torch.no_grad() for inference.
        with tf.device("/GPU:0"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Assert no NaN values
        self.assertFalse(tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values")

if __name__ == "__main__":
    tf.test.main()