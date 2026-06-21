import torch

def test_quantized_norm():
    # Create a quantized tensor
    quant_input = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), scale=1.0, zero_point=0, dtype=torch.quint8)
    
    # Dequantize the tensor before calculating the norm
    # because torch.norm does not support quantized dtypes directly.
    out_tensor = torch.norm(quant_input.dequantize())
    
    print("Quantized input:", quant_input)
    print("Output tensor:", out_tensor)

if __name__ == "__main__":
    test_quantized_norm()