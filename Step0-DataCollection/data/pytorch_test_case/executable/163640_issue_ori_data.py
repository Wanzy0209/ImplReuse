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

def main():
    torch.manual_seed(0)
    m = TinyEnc().eval()

    B, T, C = 1, 41, 512
    x = torch.randn(B, T, C, dtype=torch.float32)
    pad_mask = (torch.rand(B, T) > 0.5)
    pad_mask[..., 0] = True

    # Eager is fine
    with torch.inference_mode():
        y = m(x, pad_mask)
    print("eager ok:", tuple(y.shape))

    # Compile (fullgraph=True required to reproduce)
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    _ = cm(x, pad_mask)

if __name__ == "__main__":
    main()