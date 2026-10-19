```python
import tensorflow as tf

# Conversion: @torch.compile(fullgraph=True) -> @tf.function
@tf.function
def fn(encoder_attention_mask, encoder_hidden_states):
    # Conversion: encoder_hidden_states.new_zeros([1, 512, 3096]) -> tf.zeros with explicit dtype
    encoder_hidden_states = tf.zeros([1, 512, 3096], dtype=encoder_hidden_states.dtype)
    
    # Conversion: .sum().item() -> tf.reduce_sum()
    # Note: In TensorFlow, this returns a scalar Tensor (int32/int64), which works for slicing.
    text_len = tf.reduce_sum(encoder_attention_mask)
    
    encoder_hidden_states = encoder_hidden_states[:, :text_len]

# Conversion: torch._dynamo.config.capture_scalar_outputs is specific to PyTorch Dynamo.
# There is no direct equivalent in TensorFlow, so this line is omitted.
# mask = (torch.arange(512) < 8).unsqueeze(0).cuda()
# Conversion: torch.arange -> tf.range, unsqueeze -> expand_dims, .cuda() handled by TF device placement
mask = tf.expand_dims(tf.range(512) < 8, axis=0)

# hidden = torch.randn((1, 512, 4096)).cuda()
# Conversion: torch.randn -> tf.random.normal
hidden = tf.random.normal((1, 512, 4096))

fn(mask, hidden)
```