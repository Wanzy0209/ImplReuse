```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_isolated_language_model_mps(device):
    # Conversion: Map PyTorch device strings to TensorFlow device strings
    # "mps" (Metal Performance Shaders) is specific to PyTorch/macOS.
    # In TensorFlow, we map this to the GPU device for testing purposes.
    tf_device = "/CPU:0" if device == "cpu" else "/GPU:0"

    with tf.device(tf_device):
        # Conversion: Use TFAutoModelForCausalLM for TensorFlow
        model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")
        
        # Conversion: torch.tensor -> tf.constant
        input_ids = tf.constant([[0, 1, 0, 0], [0, 1, 2, 3]])

        attention_mask = tf.constant([
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]]],
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [ True,  True,  True, False],
              [ True,  True,  True,  True]]]])
        # attention_mask = tf.constant([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=tf.bool)

        # Conversion: torch.no_grad is not required in TensorFlow for inference
        # as gradients are only calculated within tf.GradientTape.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Conversion: torch.isnan -> tf.math.is_nan, .any() -> tf.reduce_any
        assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```