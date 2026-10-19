```python
import tensorflow as tf

class TinyEnc(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Conversion: PyTorch TransformerEncoderLayer/Encoder -> tf.keras.layers.TransformerEncoder
        # PyTorch d_model is the input dimension. TF uses key_dim (dim per head).
        # key_dim = d_model / nhead = 512 / 8 = 64.
        # PyTorch default dim_feedforward is 4 * d_model = 2048.
        key_dim = d_model // nhead
        self.enc = tf.keras.layers.TransformerEncoder(
            intermediate_dim=2048, # Matches PyTorch default dim_feedforward
            num_heads=nhead,
            num_layers=num_layers,
            dropout=0.1,
            key_dim=key_dim
        )
        # Conversion: nn.Linear -> Dense
        self.proj = tf.keras.layers.Dense(10)

    def call(self, x, pad_mask, training=None):
        # Conversion: PyTorch src_key_padding_mask uses True for padding (ignore).
        # TF attention_mask uses 1/True for valid tokens, 0/False for padding.
        # We invert the mask logic.
        attention_mask = tf.cast(~pad_mask, tf.float32)

        # Conversion: PyTorch TransformerEncoder forward call
        # TF TransformerEncoder handles the mask expansion internally
        y = self.enc(x, mask=attention_mask)
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

    # Conversion: Tensor assignment in TF requires scatter operations or variables
    # pad_mask[..., 0] = True
    indices = tf.constant([[0, 0]])
    updates = tf.constant([True])
    pad_mask = tf.tensor_scatter_nd_update(pad_mask, indices, updates)

    # Eager is fine
    # Conversion: torch.inference_mode is implicit in TF when not using GradientTape
    # We pass training=False to match eval() behavior (disable dropout)
    y = m(x, pad_mask, training=False)
    print("eager ok:", tuple(y.shape))

    # Compile
    # Conversion: torch.compile -> tf.function
    # tf.function traces the python function to a graph
    cm = tf.function(m)
    _ = cm(x, pad_mask, training=False)

if __name__ == "__main__":
    main()
```