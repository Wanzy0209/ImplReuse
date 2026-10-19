```python
import tensorflow as tf

# Conversion: @torch.compile(fullgraph=True) -> @tf.function
@tf.function
def fn(encoder_attention_mask, encoder_hidden_states):
    # Conversion: encoder_hidden_states.new_zeros([1, 512, 3072])
    # Use tf.zeros_like to preserve dtype and device, overriding shape
    encoder_hidden_states = tf.zeros_like(encoder_hidden_states, shape=[1, 512, 3072])

    # Conversion: encoder_attention_mask.sum().item()
    # In TensorFlow, we use tf.reduce_sum. We keep it as a tensor to support
    # dynamic slicing within the graph, rather than extracting a Python scalar.
    text_len = tf.reduce_sum(encoder_attention_mask)

    # Slicing operation
    encoder_hidden_states = encoder_hidden_states[:, :text_len]

# Conversion: torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent needed in TensorFlow; tf.function handles scalar outputs automatically.

# Conversion: .cuda() -> tf.device('/GPU:0')
with tf.device('/GPU:0'):
    # Conversion: torch.arange(512) -> tf.range(512)
    # Conversion: .unsqueeze(0) -> tf.expand_dims(..., axis=0)
    mask = tf.expand_dims(tf.range(512) < 8, axis=0)

    # Conversion: torch.randn((1, 512, 4096)) -> tf.random.normal((1, 512, 4096))
    hidden = tf.random.normal((1, 512, 4096))

    fn(mask, hidden)
```