import torch
import torch.nn as nn
import unittest

# Model definition from the original bug report
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

class TestTransformerEncoderCompile(unittest.TestCase):
    def test_boolean_mask_fullgraph(self):
        """
        Test that TransformerEncoder with boolean src_key_padding_mask works
        under torch.compile(fullgraph=True).
        """
        # We run the test on CPU
        device = 'cpu'
        
        torch.manual_seed(0)
        m = TinyEnc().to(device).eval()

        B, T, C = 1, 41, 512
        x = torch.randn(B, T, C, dtype=torch.float32, device=device)
        pad_mask = (torch.rand(B, T) > 0.5).to(device)
        pad_mask[..., 0] = True

        # Eager mode check
        with torch.inference_mode():
            y_eager = m(x, pad_mask)
        
        self.assertIsInstance(y_eager, torch.Tensor)
        self.assertEqual(y_eager.shape, (B, T, 10))
        print("Eager mode ok:", tuple(y_eager.shape))

        # Compile mode check (fullgraph=True required to reproduce)
        # The bug was that this raised torch._dynamo.exc.Unsupported
        try:
            cm = torch.compile(m, backend="inductor", fullgraph=True)
            with torch.inference_mode():
                y_compiled = cm(x, pad_mask)
            
            self.assertIsInstance(y_compiled, torch.Tensor)
            self.assertEqual(y_compiled.shape, (B, T, 10))
            print("Compile mode ok:", tuple(y_compiled.shape))
            
        except Exception as e:
            self.fail(f"torch.compile failed with fullgraph=True: {e}")

if __name__ == "__main__":
    unittest.main()