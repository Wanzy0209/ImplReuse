import torch
import torch.nn as nn
import torch.backends.cuda

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

def test_transformer_encoder_compile_with_padding():
    """
    Test that TransformerEncoder compiles correctly with boolean src_key_padding_mask
    under fullgraph mode, checking the state of math_sdp_enabled as it relates to
    the backend optimizations.
    """
    torch.manual_seed(0)
    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    x = torch.randn(B, T, C, dtype=torch.float32)
    pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Leverage the similar API: torch.backends.cuda.math_sdp_enabled
    # This API returns a boolean flag indicating the state of math SDP optimizations.
    # We check this to ensure the test environment is aware of the backend state
    # which might influence the code path taken by the TransformerEncoder.
    if torch.cuda.is_available():
        is_math_sdp_enabled = torch.backends.cuda.math_sdp_enabled()
        print(f"Math SDP Enabled: {is_math_sdp_enabled}")
    else:
        print("CUDA not available, running on CPU")

    # Eager mode execution
    with torch.inference_mode():
        y_eager = m(x, pad_mask)
    
    assert isinstance(y_eager, torch.Tensor), "Eager output should be a Tensor"
    assert y_eager.shape == (B, T, 10), f"Eager output shape mismatch: {y_eager.shape}"
    print("eager ok:", tuple(y_eager.shape))

    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile is not available (requires PyTorch 2.0+). Skipping compilation test.")
        return

    # Compile with fullgraph=True to reproduce the bug scenario
    # The bug was that torch._nested_tensor_from_mask_left_aligned returned a bool
    # instead of a Tensor, causing torch._dynamo.exc.Unsupported.
    try:
        cm = torch.compile(m, backend="inductor", fullgraph=True)
        with torch.inference_mode():
            y_compiled = cm(x, pad_mask)
        
        assert isinstance(y_compiled, torch.Tensor), "Compiled output must be a Tensor, not bool"
        assert y_compiled.shape == (B, T, 10), f"Compiled output shape mismatch: {y_compiled.shape}"
        print("compile ok:", tuple(y_compiled.shape))
        
    except Exception as e:
        print(f"Compilation failed with error: {e}")
        raise

if __name__ == "__main__":
    test_transformer_encoder_compile_with_padding()