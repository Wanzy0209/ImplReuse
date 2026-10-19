```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

# Conversion: PyTorch "mps" (Metal Performance Shaders) is specific to Apple Silicon.
# In TensorFlow, this is typically handled by the standard "gpu" device.
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_isolated_language_model_mps(device):
    with tf.device(device):
        model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")
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

        # Conversion: TensorFlow does not track gradients in eager execution by default,
        # so a torch.no_grad() equivalent context is not required.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```