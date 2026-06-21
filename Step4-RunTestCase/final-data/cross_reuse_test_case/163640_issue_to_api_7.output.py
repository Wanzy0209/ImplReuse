import torch
import torch.nn as nn
import torch.library

# Define a custom operator to leverage torch.library.get_ctx
# This mimics the pattern of registering a fake implementation,
# which is relevant to the underlying issue of handling internal ops
# like _nested_tensor_from_mask_left_aligned during compilation.

def custom_mask_op(mask: torch.Tensor) -> torch.Tensor:
    return mask

def custom_mask_op_fake(mask: torch.Tensor) -> torch.Tensor:
    # Leverage the similar API: torch.library.get_ctx
    # This is required to be inside a register_fake block
    # Check for availability to support older PyTorch versions
    if hasattr(torch.library, "get_ctx"):
        ctx = torch.library.get_ctx()
    # Return a tensor with the correct shape and dtype for the fake context
    return torch.empty_like(mask)

# Register the custom operator
torch.library.define("test_lib::custom_mask_op", "(Tensor) -> Tensor")
torch.library.impl("test_lib::custom_mask_op", custom_mask_op)

# Handle API differences: PyTorch 2.1+ uses register_fake,
# older versions (e.g., 2.0) use impl with dispatch_key="Meta"
if hasattr(torch.library, "register_fake"):
    torch.library.register_fake("test_lib::custom_mask_op", custom_mask_op_fake)
else:
    torch.library.impl("test_lib::custom_mask_op", custom_mask_op_fake, dispatch_key="Meta")

class TinyEnc(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, 10)

    def forward(self, x, pad_mask):
        # Apply the custom operator that uses the similar API pattern
        processed_mask = torch.ops.test_lib.custom_mask_op(pad_mask)
        
        # Passing src_key_padding_mask triggers torch._nested_tensor_from_mask_left_aligned
        y = self.enc(x, mask=None, src_key_padding_mask=processed_mask)
        return self.proj(y)

def main():
    torch.manual_seed(0)
    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    x = torch.randn(B, T, C, dtype=torch.float32)
    pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Eager execution
    with torch.inference_mode():
        y_eager = m(x, pad_mask)
    print("eager ok:", tuple(y_eager.shape))

    # Compile with fullgraph=True to reproduce the original issue context
    # This tests if the TransformerEncoder works with padding masks in fullgraph
    # while also ensuring the custom op using get_ctx integrates correctly.
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        y_compiled = cm(x, pad_mask)
    
    print("compile ok:", tuple(y_compiled.shape))

    # Verify results match
    assert torch.allclose(y_eager, y_compiled, atol=1e-2)

if __name__ == "__main__":
    main()