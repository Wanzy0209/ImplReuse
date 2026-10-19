```python
import tensorflow as tf

# Conversion: nn.Module -> tf.keras.Model
class TinyEnc(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: nn.TransformerEncoderLayer -> tf.keras.layers.TransformerEncoderLayer
        # Note: TF uses 'num_heads' instead of 'nhead' and 'dropout_rate' instead of 'dropout'.
        # 'batch_first' is default behavior in TF Keras (inputs are (batch, seq, feature)).
        layer = tf.keras.layers.TransformerEncoderLayer(
            d_model=d_model, 
            num_heads=nhead, 
            dropout_rate=0.1
        )
        # Conversion: nn.TransformerEncoder -> tf.keras.layers.TransformerEncoder
        self.enc = tf.keras.layers.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.proj = tf.keras.layers.Dense(10)

    # Conversion: forward -> call
    def call(self, x, pad_mask, training=False):
        # Conversion: src_key_padding_mask -> padding_mask
        # Passing padding_mask triggers internal masking logic in TF Transformer
        y = self.enc(x, padding_mask=pad_mask, training=training)
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
    
    # Conversion: pad_mask[..., 0] = True
    # TF tensors are immutable, so we use tensor_scatter_nd_update to modify specific indices.
    indices = tf.constant([[i, 0] for i in range(B)])
    updates = tf.constant([True] * B, dtype=tf.bool)
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode -> No direct equivalent context manager needed for simple forward pass,
    # but we ensure training=False is passed to layers.
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    # tf.function compiles the callable into a static graph for performance.
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```