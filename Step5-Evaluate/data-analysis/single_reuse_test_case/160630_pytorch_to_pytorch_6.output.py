import torch

def test_quantized_quantile():
    # Create a quantized tensor
    quant_input = torch.quantize_per_tensor(torch.tensor([1.0, 2.0, 3.0]), scale=1.0, zero_point=0, dtype=torch.quint8)
    
    # Attempt to calculate the quantile (median) of the quantized tensor
    # torch.quantile does not support quantized tensors directly (QUInt8),
    # so we must dequantize the tensor to a floating point tensor first.
    out_tensor = torch.quantile(quant_input.dequantize(), 0.5)
    
    print("Quantized input:", quant_input)
    print("Output tensor:", out_tensor)

if __name__ == "__main__":
    test_quantized_quantile()