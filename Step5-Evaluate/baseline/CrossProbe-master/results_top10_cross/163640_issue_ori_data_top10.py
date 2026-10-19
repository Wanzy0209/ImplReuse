```python
import tensorflow as tf

class TinyEnc(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: nn.TransformerEncoderLayer -> tf.keras.layers.TransformerEncoderLayer
        # Note: TF layers default to batch_first=True behavior.
        layer = tf.keras.layers.TransformerEncoderLayer(
            d_model=d_model, num_heads=nhead, dropout=0.1
        )
        # Conversion: nn.TransformerEncoder -> tf.keras.layers.TransformerEncoder
        self.enc = tf.keras.layers.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.proj = tf.keras.layers.Dense(10)

    def call(self, x, pad_mask):
        # Passing padding_mask triggers masking logic
        # Conversion: src_key_padding_mask -> padding_mask
        y = self.enc(x, padding_mask=pad_mask)
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
    # Conversion: pad_mask[..., 0] = True -> tf.tensor_scatter_nd_update
    # Tensors are immutable in TF, so we use scatter update
    indices = tf.constant([[i, 0] for i in range(B)])
    updates = tf.constant([True] * B, dtype=tf.bool)
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode -> No direct equivalent needed for inference-only pass
    # Gradients are not calculated outside of tf.GradientTape
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    # tf.function compiles the callable into a graph
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```