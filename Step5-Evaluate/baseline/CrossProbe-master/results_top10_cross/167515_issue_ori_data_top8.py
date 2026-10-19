```python
import pytest
import tensorflow as tf
from transformers import TFAutoModelForCausalLM

# Conversion note: PyTorch 'mps' device is specific to Apple Silicon.
# TensorFlow typically uses 'cpu' and 'gpu'. We map 'mps' to 'gpu' for validity.
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_isolated_language_model_mps(device):
    # Conversion note: TF models are loaded without .to(device).
    # Device placement is handled via tf.device context manager.
    model = TFAutoModelForCausalLM.from_pretrained("sbintuitions/tiny-lm")

    with tf.device(device):
        # Conversion note: torch.tensor -> tf.constant
        input_ids = tf.constant([[0, 1, 0, 0], [0, 1, 2, 3]])

        attention_mask = tf.constant([
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [False, False, False, False],
              [False, False, False, False]]],
            [[[ True, False, False, False],
              [ True,  True, False, False],
              [ True,  True,  True, False],
              [ True,  True,  True,  True]]]
        ], dtype=tf.bool)
        # attention_mask = tf.constant([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=tf.bool)

        # Conversion note: torch.no_grad is not needed in TF for inference
        # as gradients are only tracked inside tf.GradientTape.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Conversion note: torch.isnan(...).any() -> tf.reduce_any(tf.math.is_nan(...))
        assert not tf.reduce_any(tf.math.is_nan(outputs.logits)), "Logits contain NaN values"
```