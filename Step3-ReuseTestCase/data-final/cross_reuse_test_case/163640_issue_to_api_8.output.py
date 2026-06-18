import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Leveraging the similar API: torch.nn.functional.tanh
        # This integrates the pattern of calling functional APIs within the compiled graph,
        # ensuring that type handling (Tensor vs bool) is consistent across ops.
        y = F.tanh(y)
        
        return self.proj(y)

def test_transformer_encoder_with_boolean_mask_and_tanh():
    torch.manual_seed(0)
    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    x = torch.randn(B, T, C, dtype=torch.float32)
    pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Eager mode check
    with torch.inference_mode():
        y_eager = m(x, pad_mask)
    assert y_eager.shape == (B, T, 10), "Eager mode output shape mismatch"
    print("eager ok:", tuple(y_eager.shape))

    # Compile mode check (fullgraph=True required to reproduce the bug)
    # The bug causes torch._nested_tensor_from_mask_left_aligned to return a bool,
    # breaking the graph. This test verifies that the graph handles the mask correctly
    # alongside standard functional ops like tanh.
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        y_compiled = cm(x, pad_mask)
    
    assert y_compiled.shape == (B, T, 10), "Compiled mode output shape mismatch"
    # Verify numerical consistency
    assert torch.allclose(y_eager, y_compiled, atol=1e-4), "Output mismatch between eager and compiled"
    print("compile ok:", tuple(y_compiled.shape))

if __name__ == "__main__":
    test_transformer_encoder_with_boolean_mask_and_tanh()