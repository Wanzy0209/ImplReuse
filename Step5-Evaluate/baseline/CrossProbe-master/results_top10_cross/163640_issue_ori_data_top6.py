```python
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras

class TinyEnc(keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super(TinyEnc, self).__init__()
        # Conversion: nn.TransformerEncoderLayer -> tf.keras.layers.TransformerEncoderLayer
        # Note: batch_first is default behavior in TF Keras
        layer = layers.TransformerEncoderLayer(
            d_model=d_model, num_heads=nhead, dropout=0.1
        )
        # Conversion: nn.TransformerEncoder -> tf.keras.layers.TransformerEncoder
        self.enc = layers.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.proj = layers.Dense(10)

    def call(self, x, pad_mask):
        # Conversion: forward -> call
        # Passing mask (equivalent to src_key_padding_mask)
        # training=False to disable dropout (mimics .eval() and inference_mode)
        y = self.enc(x, mask=pad_mask, training=False)
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
    # Conversion: In-place tensor update -> tf.tensor_scatter_nd_update
    # pad_mask[..., 0] = True
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, [[0, 0]], [True])

    # Eager is fine
    # Conversion: torch.inference_mode -> explicit training=False in call
    y = m(x, pad_mask, training=False)
    print("eager ok:", y.shape)

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    # tf.function traces the whole graph by default
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```