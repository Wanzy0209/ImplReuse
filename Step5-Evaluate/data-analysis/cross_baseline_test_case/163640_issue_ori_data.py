```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Conversion: nn.Module -> tf.keras.Model
class TinyEnc(Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: nn.TransformerEncoderLayer -> tf.keras.layers.TransformerEncoderLayer
        # Note: batch_first is default in TF Keras (batch, seq, feature)
        layer = layers.TransformerEncoderLayer(
            embed_dim=d_model, num_heads=nhead, dropout=0.1
        )
        # Conversion: nn.TransformerEncoder -> tf.keras.layers.TransformerEncoder
        self.enc = layers.TransformerEncoder(layer, num_layers=num_layers)
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.proj = layers.Dense(10)

    # Conversion: forward -> call
    def call(self, x, pad_mask, training=None):
        # Passing mask triggers masking in TF Transformer
        # Note: TF expects mask to be boolean where True = padding
        y = self.enc(x, mask=pad_mask, training=training)
        return self.proj(y)

def main():
    # Conversion: torch.manual_seed -> tf.random.set_seed
    tf.random.set_seed(0)
    m = TinyEnc()

    B, T, C = 1, 41, 512
    # Conversion: torch.randn -> tf.random.normal
    x = tf.random.normal((B, T, C), dtype=tf.float32)
    
    # Conversion: torch.rand -> tf.random.uniform
    # pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask = tf.random.uniform((B, T)) > 0.5
    
    # Conversion: pad_mask[..., 0] = True
    # TF tensors are immutable, so we use scatter_nd_update
    indices = tf.constant([[i, 0] for i in range(B)])
    updates = tf.constant([True] * B, dtype=tf.bool)
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode -> training=False
    # In TF, we pass training=False to the model to disable dropout/batchnorm updates
    y = m(x, pad_mask, training=False)
    print("eager ok:", y.shape)

    # Compile (fullgraph=True required to reproduce)
    # Conversion: torch.compile -> tf.function
    # tf.function creates a graph, similar to torch.compile
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```