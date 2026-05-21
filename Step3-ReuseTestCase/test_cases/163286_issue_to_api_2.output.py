import torch
import torch.nn as nn
from torch.testing import assert_close

def test_transformer_encoder_preserves_dtype_view():
    """
    Test that torch.compile (Inductor) preserves .view(dtype) operations
    through as_strided (transpose) operations when used with TransformerEncoder.
    
    This relates to issue #163286 where as_strided lowering was throwing away
    the .view(dtype), causing incorrect dtype handling in kernels.
    
    The test constructs a scenario where a tensor is viewed as a different dtype,
    transposed (which lowers to as_strided), and viewed back. If the compiler
    incorrectly lowers this by discarding the intermediate view, the numerical
    output will differ from the eager execution.
    """
    
    class ViewTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Initialize a standard TransformerEncoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        def forward(self, x):
            # Input shape: (Seq=10, Batch=32, Dim=128)
            
            # 1. View as uint8. This changes the size of the last dimension (128 * 4 bytes = 512).
            # This mimics the 'output_scales_ptr' being viewed as uint8 in the bug report.
            x_uint8 = x.view(torch.uint8) # Shape: (10, 32, 512)
            
            # 2. Transpose the tensor. In PyTorch, transpose is implemented via as_strided.
            # This is the operation identified in the bug report where the view was being discarded.
            x_transposed = x_uint8.transpose(0, 1) # Shape: (32, 10, 512)
            
            # 3. View back to float32.
            x_float = x_transposed.view(torch.float32) # Shape: (32, 10, 128)
            
            # 4. Pass to TransformerEncoder.
            # The encoder handles (Seq, Batch, Dim). Here we pass (32, 10, 128),
            # so it treats Batch=32 as Seq and Seq=10 as Batch.
            return self.encoder(x_float)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViewTransformerModel().to(device)
    
    # Compile the model using Inductor (default backend)
    compiled_model = torch.compile(model)
    
    # Create random input
    inp = torch.randn(10, 32, 128, device=device)
    
    # Run eager mode
    model.eval()
    with torch.no_grad():
        out_eager = model(inp)
    
    # Run compiled mode
    compiled_model.eval()
    with torch.no_grad():
        out_compiled = compiled_model(inp)
    
    # Assert that the compiled output matches the eager output.
    # If the bug exists (view thrown away during as_strided lowering),
    # the compiled model will treat the tensor as float32 during the transpose,
    # resulting in a completely different bit pattern and numerical values
    # compared to the eager mode which correctly respects the uint8 view.
    assert_close(out_compiled, out_eager, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_transformer_encoder_preserves_dtype_view()