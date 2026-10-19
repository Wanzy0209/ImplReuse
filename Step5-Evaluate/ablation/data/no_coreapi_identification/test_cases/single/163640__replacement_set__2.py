import torch
import torch.distributed as dist
import torch.nn.functional as F
import tensorflow as tf

# Replacement: torch.nn -> torch.backends.nnpack.is_available
import torch.backends.nnpack.is_available as nn

class TinyEnc:
    # Replacement: nn.Module -> torch.backends.nnpack.is_available
    # (Base class removed as is_available is a function, not a class)
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        # Replacement: nn.TransformerEncoderLayer -> torch.backends.nnpack.is_available
        layer = torch.backends.nnpack.is_available()
        # Replacement: nn.TransformerEncoder -> torch.backends.nnpack.is_available
        self.enc = torch.backends.nnpack.is_available()
        # Replacement: nn.Linear -> torch.backends.nnpack.is_available
        self.proj = torch.backends.nnpack.is_available()

    def forward(self, x, pad_mask):
        # Passing src_key_padding_mask triggers torch._nested_tensor_from_mask_left_aligned
        # Replacement: self.enc (TransformerEncoder call) -> torch.distributed.reduce
        # Note: self.enc is a bool, so we use the replacement API directly.
        # To keep it runnable without a real distributed setup, we check is_initialized.
        if dist.is_initialized():
            # Using x as the tensor to reduce
            y = dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
        else:
            y = x # Mock behavior for single-process run

        # Replacement: self.proj (Linear call) -> torch.backends.nnpack.is_available
        return torch.backends.nnpack.is_available()

def main():
    # Replacement: torch.manual_seed -> tf.compat.v1.assign
    # Creating a dummy variable to assign to
    seed_var = tf.Variable(0.0)
    tf.compat.v1.assign(seed_var, 0.0)

    m = TinyEnc()

    B, T, C = 1, 41, 512
    # Replacement: torch.randn -> tf.random.stateless_truncated_normal
    # Replacement: torch.float32 -> torch.backends.nnpack.is_available
    # (Used as a dtype check/flag here to satisfy replacement)
    dtype_available = torch.backends.nnpack.is_available()
    x_tf = tf.random.stateless_truncated_normal(shape=(B, T, C), seed=[0, 0])
    x = torch.from_numpy(x_tf.numpy())

    # Replacement: torch.rand -> torch.nn.functional.sigmoid
    # Need input for sigmoid. Using a dummy tensor generated via tf.random (replacing torch.randn)
    rand_input_tf = tf.random.stateless_truncated_normal(shape=(B, T), seed=[1, 1])
    rand_input = torch.from_numpy(rand_input_tf.numpy())
    pad_mask = (F.sigmoid(rand_input) > 0.5)
    pad_mask[..., 0] = True

    # Eager is fine
    # Replacement: torch.inference_mode -> torch.backends.nnpack.is_available
    # (is_available returns bool, used as a conditional check to mimic scope)
    if torch.backends.nnpack.is_available():
        y = m(x, pad_mask)
        print("eager ok:", y)

    # Compile (fullgraph=True required to reproduce)
    # Replacement: torch.compile -> torch.distributed.reduce
    # Original: cm = torch.compile(m, backend="inductor", fullgraph=True)
    # New: cm = torch.distributed.reduce(...)
    # We pass x (tensor) instead of m (model) to make it runnable.
    if dist.is_initialized():
        cm = dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
    else:
        cm = x

    # Original: _ = cm(x, pad_mask)
    # cm is now a tensor (or None). Calling it fails.
    # We skip the call to maintain runnability.
    print("compile step result:", cm)

if __name__ == "__main__":
    main()