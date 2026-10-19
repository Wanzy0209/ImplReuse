```python
import tensorflow as tf
import pytest
from transformers import TFAutoModelForCausalLM

# Conversion: PyTorch uses "mps" for Apple Silicon. TensorFlow uses "GPU" generally.
# Updated parametrize to valid TensorFlow device strings.
@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_isolated_language_model_mps(device):
    # Conversion: AutoModelForCausalLM -> TFAutoModelForCausalLM
    # Conversion: .to(device) -> with tf.device(device):
    with tf.device(device):
        model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")

        # Conversion: torch.tensor -> tf.constant
        input_ids = tf.constant([[0, 1, 0, 0], [0, 1, 2, 3]], dtype=tf.int32)

        attention_mask = tf.constant([
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]]],
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [ True,  True,  True, False],
              [ True,  True,  True,  True]]]], dtype=tf.bool)
        # attention_mask = tf.constant([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=tf.bool)

        # Conversion: torch.no_grad is not required in TensorFlow eager execution.
        # Gradients are only calculated within tf.GradientTape contexts.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Conversion: torch.isnan(...).any() -> tf.reduce_any(tf.math.is_nan(...))
        assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```