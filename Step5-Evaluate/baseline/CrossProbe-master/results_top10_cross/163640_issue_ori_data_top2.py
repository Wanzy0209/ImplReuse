```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TinyEnc(keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: nn.TransformerEncoderLayer -> layers.TransformerEncoderLayer
        # Note: TF uses 'num_heads' instead of 'nhead'. 
        # TF defaults to batch_first=True for 3D inputs.
        layer = layers.TransformerEncoderLayer(
            d_model=d_model, num_heads=nhead, dropout=0.1
        )
        # Conversion: nn.TransformerEncoder -> layers.TransformerEncoder
        self.enc = layers.TransformerEncoder(layer, num_layers=num_layers)
        # Conversion: nn.Linear -> layers.Dense
        self.proj = layers.Dense(10)

    def call(self, x, pad_mask, training=None):
        # Passing padding_mask triggers masking logic in TF Transformer
        # Conversion: src_key_padding_mask -> padding_mask
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
    pad_mask = (tf.random.uniform((B, T)) > 0.5)
    
    # Conversion: In-place assignment -> tensor_scatter_nd_update
    # pad_mask[..., 0] = True
    indices = tf.constant([[i, 0] for i in range(B)])
    updates = tf.constant([True] * B, dtype=tf.bool)
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode -> training=False argument
    # TF does not track gradients outside of GradientTape, so explicit context manager is not needed.
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    # tf.function traces the python function into a static graph, similar to torch.compile
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```