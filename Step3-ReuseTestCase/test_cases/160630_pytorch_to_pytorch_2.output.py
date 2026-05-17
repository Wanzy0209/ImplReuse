import torch
import torch.nn as nn

def test_lazy_linear_quantized():
    # Create a quantized tensor
    quant_input = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), scale=1.0, zero_point=0, dtype=torch.quint8)
    
    # Initialize LazyLinear
    # LazyLinear infers in_features from the input size on the first forward pass
    lazy_linear = nn.LazyLinear(out_features=10)
    
    # Attempt to run the layer with quantized input
    # This tests if LazyLinear handles quantized inputs or fails similarly to zeros_like
    try:
        out_tensor = lazy_linear(quant_input)
        print("Quantized input:", quant_input)
        print("Output tensor:", out_tensor)
    except Exception as e:
        print(f"Error encountered: {e}")

if __name__ == "__main__":
    test_lazy_linear_quantized()