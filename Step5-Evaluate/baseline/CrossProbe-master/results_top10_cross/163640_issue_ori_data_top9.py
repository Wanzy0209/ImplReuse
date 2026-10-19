```python
import tensorflow as tf

class TinyEnc(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: nn.TransformerEncoderLayer -> tf.keras.layers.TransformerEncoderLayer
        # Note: batch_first is default in TF Keras
        layer = tf.keras.layers.TransformerEncoderLayer(
            num_heads=nhead,
            d_model=d_model,
            dropout=0.1
        )
        # Conversion: nn.TransformerEncoder -> tf.keras.layers.TransformerEncoder
        self.enc = tf.keras.layers.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.proj = tf.keras.layers.Dense(10)

    def call(self, x, pad_mask, training=False):
        # Conversion: PyTorch src_key_padding_mask uses True for padding.
        # TF attention_mask uses 0 for padding (1 for valid).
        # We invert the boolean mask and cast to int.
        attention_mask = tf.cast(~pad_mask, tf.int32)

        # Passing src_key_padding_mask triggers torch._nested_tensor_from_mask_left_aligned
        # TF handles padding via attention_mask
        y = self.enc(x, attention_mask=attention_mask, training=training)
        return self.proj(y)

def main():
    # Conversion: torch.manual_seed -> tf.random.set_seed
    tf.random.set_seed(0)
    m = TinyEnc()

    B, T, C = 1, 41, 512
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal((B, T, C), dtype=tf.float32)
    # Conversion: torch.rand -> tf.random.uniform
    pad_mask = tf.random.uniform((B, T)) > 0.5
    
    # Conversion: Tensor assignment pad_mask[..., 0] = True
    # TF tensors are immutable, use scatter_nd_update
    indices = tf.constant([[i, 0] for i in range(B)])
    updates = tf.constant([True] * B, dtype=tf.bool)
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode is implicit in TF eager for gradients, 
    # but we pass training=False to disable dropout/batchnorm updates.
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```