```python
import tensorflow as tf

# Attempt to create a zeroed tensor with the same shape
# Conversion: torch.zeros_like -> tf.zeros_like
out_tensor = tf.zeros_like(quant_input)  # <-- This triggers the error

print("Quantized input:", quant_input)
print("Output tensor:", out_tensor)
```