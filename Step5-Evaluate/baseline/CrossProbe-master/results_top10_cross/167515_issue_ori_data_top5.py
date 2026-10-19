```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_isolated_language_model_mps(device):
    # Conversion: Map PyTorch device strings to TensorFlow device strings
    # 'mps' in PyTorch typically maps to '/GPU:0' in TensorFlow (when using MPS backend)
    tf_device = "/CPU:0" if device == "cpu" else "/GPU:0"

    with tf.device(tf_device):
        # Conversion: AutoModelForCausalLM -> TFAutoModelForCausalLM
        model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")

        # Conversion: torch.tensor -> tf.constant
        # Note: Device placement is handled by the outer tf.device context
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

        # Conversion: torch.no_grad is not required in TensorFlow 2.x eager execution
        # Gradients are only tracked within tf.GradientTape
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Conversion: torch.isnan(x).any() -> tf.reduce_any(tf.math.is_nan(x))
        # .numpy() is used to retrieve the boolean value from the tensor for the assertion
        assert not tf.reduce_any(tf.math.is_nan(outputs.logits)).numpy(), "Logits contain NaN values"
```