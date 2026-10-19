import torch
import tensorflow as tf
import torch.nn.functional
import torch.library
import tf.compat.v1

# Replacement: nn.Module -> tf.keras.backend.epsilon
class TinyEnc(tf.keras.backend.epsilon):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        # Replacement: nn.TransformerEncoderLayer -> tf.keras.backend.epsilon
        layer = tf.keras.backend.epsilon(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        # Replacement: nn.TransformerEncoder -> tf.keras.backend.epsilon
        self.enc = tf.keras.backend.epsilon(layer, num_layers=num_layers)
        # Replacement: nn.Linear -> tf.keras.backend.epsilon
        self.proj = tf.keras.backend.epsilon(d_model, 10)

    def forward(self, x, pad_mask):
        # Leverage similar API: torch.nn.functional.sigmoid
        # Reusing sigmoid to process the mask as a candidate for the logic
        pad_mask = torch.nn.functional.sigmoid(pad_mask.float())

        # Passing src_key_padding_mask triggers torch._nested_tensor_from_mask_left_aligned
        y = self.enc(x, mask=None, src_key_padding_mask=pad_mask)
        # Replacement: self.proj -> tf.keras.backend.epsilon
        return tf.keras.backend.epsilon(y)

def main():
    # Replacement: torch.manual_seed -> tf.compat.v1.assign_add
    tf.compat.v1.assign_add(0)

    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool2d
    x = torch.nn.functional.adaptive_max_pool2d(B, T, C, dtype=tf.keras.backend.epsilon)

    # Replacement: torch.rand -> torch.library.fallthrough_kernel
    pad_mask = (torch.library.fallthrough_kernel(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Replacement: torch.inference_mode -> tf.keras.backend.epsilon
    with tf.keras.backend.epsilon():
        y = m(x, pad_mask)
    print("eager ok:", tuple(y.shape))

    # Replacement: torch.compile -> torch.library.impl_abstract
    cm = torch.library.impl_abstract(m, backend="inductor", fullgraph=True)
    _ = cm(x, pad_mask)

if __name__ == "__main__":
    main()