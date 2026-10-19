```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_isolated_language_model_mps(device):
    # Conversion: Map PyTorch device strings to TensorFlow device strings
    # "mps" (Metal Performance Shaders) maps to "/gpu:0" in TensorFlow on macOS
    tf_device = "/cpu:0" if device == "cpu" else "/gpu:0"

    with tf.device(tf_device):
        # Conversion: Use TFAutoModelForCausalLM for TensorFlow
        model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")

        # Conversion: torch.tensor -> tf.constant. Device placement handled by tf.device context.
        input_ids = tf.constant([[0, 1, 0, 0], [0, 1, 2, 3]])

        attention_mask = tf.constant([
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]]],
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [ True,  True,  True, False],
              [ True,  True,  True,  True]]]], dtype=tf.bool)
        # attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool, device=device)

        # Conversion: torch.no_grad is not needed in TensorFlow for inference
        # as gradients are only tracked inside tf.GradientTape.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Conversion: torch.isnan -> tf.math.is_nan, .any() -> tf.reduce_any
    assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```