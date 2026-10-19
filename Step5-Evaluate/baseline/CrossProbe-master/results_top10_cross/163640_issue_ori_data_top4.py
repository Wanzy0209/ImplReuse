```python
import tensorflow as tf

class TinyEnc(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: PyTorch TransformerEncoderLayer uses d_model (input dim) and nhead.
        # TensorFlow TransformerEncoderLayer uses key_dim (dim per head).
        # key_dim = d_model / nhead
        key_dim = d_model // nhead
        layer = tf.keras.layers.TransformerEncoderLayer(
            num_heads=nhead,
            key_dim=key_dim,
            dropout=0.1,
            # feed_forward_dim defaults to 2048 (4 * d_model), matching PyTorch default
        )
        # Conversion: PyTorch TransformerEncoder takes layer and num_layers.
        # TensorFlow TransformerEncoder takes a list of layers.
        self.enc = tf.keras.layers.TransformerEncoder([layer] * num_layers)
        self.proj = tf.keras.layers.Dense(10)

    def call(self, x, pad_mask, training=False):
        # Conversion: PyTorch src_key_padding_mask maps to TF padding_mask.
        # Both expect boolean mask where True indicates padding/masked value.
        y = self.enc(x, padding_mask=pad_mask, training=training)
        return self.proj(y)

def main():
    tf.random.set_seed(0)
    m = TinyEnc()

    B, T, C = 1, 41, 512
    x = tf.random.normal((B, T, C), dtype=tf.float32)
    pad_mask = tf.random.uniform((B, T)) > 0.5
    
    # Conversion: PyTorch slice assignment pad_mask[..., 0] = True
    # TF tensors are immutable, use scatter update
    indices = tf.constant([[i, 0] for i in range(B)])
    updates = tf.constant([True] * B, dtype=tf.bool)
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode() implies training=False in TF
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile
    # Conversion: torch.compile maps to tf.function
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```