```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

# Conversion: Changed "mps" to "gpu" as TensorFlow uses "gpu" for accelerator devices
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_isolated_language_model_mps(device):
    # Conversion: Using TFAutoModelForCausalLM for TensorFlow
    # Conversion: Removed .to(device) as TF handles device placement via context managers
    model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")

    # Conversion: torch.tensor -> tf.constant, device placement handled by tf.device context
    with tf.device(device):
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

        # Conversion: torch.no_grad is not needed in TF eager execution (gradients are not tracked by default)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Conversion: torch.isnan -> tf.math.is_nan, .any() -> tf.reduce_any
        assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```