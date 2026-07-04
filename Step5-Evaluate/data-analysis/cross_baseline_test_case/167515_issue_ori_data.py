```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_isolated_language_model_mps(device):
    # Conversion: Use TFAutoModelForCausalLM for TensorFlow
    # Conversion: Use tf.device context for device placement instead of .to(device)
    with tf.device(device):
        model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")

        # Conversion: tf.constant replaces torch.tensor
        # Note: Device placement is handled by the tf.device context
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

    # Conversion: torch.no_grad is not required in TensorFlow 2.x eager execution for inference
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Conversion: tf.math.is_nan and tf.reduce_any replace torch.isnan().any()
    assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```