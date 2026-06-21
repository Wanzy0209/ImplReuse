import torch
import torch.nn as nn

class TinyEnc(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, 10)

    def forward(self, x, pad_mask):
        # Passing src_key_padding_mask triggers torch._nested_tensor_from_mask_left_aligned
        y = self.enc(x, mask=None, src_key_padding_mask=pad_mask)
        return self.proj(y)

def test_transformer_encoder_fullgraph_with_padding_mask():
    """
    Test case for Issue 163640.
    Verifies that TransformerEncoder works with boolean src_key_padding_mask 
    under torch.compile(fullgraph=True).
    
    Also leverages torch.backends.mkldnn.is_available to verify that 
    boolean returns from C++ extensions are handled correctly in fullgraph mode.
    """
    
    # Check for torch.compile availability (PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("Skipping test: torch.compile is not available (requires PyTorch >= 2.0)")
        return

    # Reuse similar API: Verify boolean return handling in fullgraph
    # This acts as a control to ensure the compiler handles bools correctly
    # when they are expected (as opposed to the bug where a bool is returned
    # where a Tensor is expected).
    def check_backend():
        return torch.backends.mkldnn.is_available()

    compiled_check = torch.compile(check_backend, fullgraph=True)
    is_mkldnn_available = compiled_check()
    assert isinstance(is_mkldnn_available, bool), "MKLDNN check should return a boolean"

    # Original Bug Reproduction Logic
    torch.manual_seed(0)
    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    x = torch.randn(B, T, C, dtype=torch.float32)
    pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Eager execution
    with torch.inference_mode():
        y_eager = m(x, pad_mask)
    
    print(f"Eager output shape: {tuple(y_eager.shape)}")

    # Compile with fullgraph=True to trigger the specific bug path
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    
    with torch.inference_mode():
        y_compiled = cm(x, pad_mask)

    print(f"Compiled output shape: {tuple(y_compiled.shape)}")

    # Verify results match
    assert torch.allclose(y_eager, y_compiled), "Mismatch between eager and compiled outputs"
    assert y_eager.shape == (B, T, 10), "Unexpected output shape"
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_transformer_encoder_fullgraph_with_padding_mask()