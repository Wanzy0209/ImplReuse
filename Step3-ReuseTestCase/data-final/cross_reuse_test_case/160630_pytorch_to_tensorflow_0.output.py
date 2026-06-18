import tensorflow as tf

def test_quantized_triu():
    # Create a quantized tensor
    # Note: triu requires rank >= 2, so we use a 2x2 matrix instead of a 1D vector
    # PyTorch used quantize_per_tensor on floats. Here we cast to quint8 to simulate the quantized type.
    quant_input = tf.cast([[1.0, 2.0], [3.0, 4.0]], dtype=tf.quint8)
    
    # Attempt to apply triu
    # This internally creates a zero constant of the same dtype (similar to zeros_like logic)
    out_tensor = tf.experimental.numpy.triu(quant_input)
    
    print("Quantized input:", quant_input)
    print("Output tensor:", out_tensor)

if __name__ == "__main__":
    test_quantized_triu()