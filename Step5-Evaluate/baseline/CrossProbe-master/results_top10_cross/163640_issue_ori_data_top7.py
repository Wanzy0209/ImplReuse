```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Conversion: nn.Module -> keras.Model
class TinyEnc(keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: nn.TransformerEncoderLayer -> layers.TransformerEncoderLayer
        # Note: Keras defaults to batch_first=True
        layer = layers.TransformerEncoderLayer(
            d_model=d_model, num_heads=nhead, dropout=0.1
        )
        # Conversion: nn.TransformerEncoder -> layers.TransformerEncoder
        self.enc = layers.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
        # Conversion: nn.Linear -> layers.Dense
        self.proj = layers.Dense(10)

    # Conversion: forward -> call
    def call(self, x, pad_mask, training=None):
        # Passing padding_mask triggers masking in Keras Transformer
        # Conversion: src_key_padding_mask -> padding_mask
        y = self.enc(x, padding_mask=pad_mask, training=training)
        return self.proj(y)

def main():
    # Conversion: torch.manual_seed -> tf.random.set_seed
    tf.random.set_seed(0)
    m = TinyEnc()

    B, T, C = 1, 41, 512
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal(shape=(B, T, C), dtype=tf.float32)
    # Conversion: torch.rand -> tf.random.uniform
    pad_mask = (tf.random.uniform(shape=(B, T)) > 0.5)
    
    # Conversion: In-place tensor update -> tf.tensor_scatter_nd_update
    # pad_mask[..., 0] = True
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, [[0, 0]], [True])

    # Eager is fine
    # Conversion: torch.inference_mode -> training=False
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    # tf.function compiles the callable into a static graph
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```