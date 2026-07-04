# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
# Attempt to create a zeroed tensor with the same shape
out_tensor = torch.zeros_like(quant_input)  # <-- This triggers the error

print("Quantized input:", quant_input)
print("Output tensor:", out_tensor)