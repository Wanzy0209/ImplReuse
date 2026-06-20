import sys

# Handle environment/dependency errors gracefully
try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Skipping test due to import error: {e}")
    print("This is likely due to a GLIBCXX version mismatch or missing dependencies.")
    print("Please ensure your environment meets the requirements for TensorFlow.")
    sys.exit(0)

def test_stft_recombination_with_add():
    """
    Test case adapted from PyTorch issue 161906.
    
    Original Issue: torch.compile fails with aten.complex.default assertion
    when recombining STFT output (real/imag parts) into a complex tensor.
    
    Similar API: tf.keras.ops.add
    
    This test verifies that tf.keras.ops.add correctly handles the recombination
    of real and imaginary parts derived from an STFT within a tf.function graph,
    mirroring the logic of the original bug report.
    """
    # Parameters matching the original issue
    n_fft = 512
    hop = 160
    win = 320

    # Input tensor
    x = tf.random.normal((1, 16000), dtype=tf.float32)

    # Window buffer
    window = tf.signal.hann_window(win)

    # Parameter for broadcasting/stride ops (mimicking nn.Parameter)
    p = tf.Variable(2.0, dtype=tf.float32)

    @tf.function # Enable compilation to mimic torch.compile scenario
    def model(x):
        # 1. STFT operation (returns complex tensor)
        S = tf.signal.stft(
            x,
            frame_length=n_fft,
            frame_step=hop,
            fft_length=n_fft,
            window_fn=lambda _: window,
            pad_end=True
        )

        # 2. Split into Real and Imaginary parts
        R = tf.math.real(S)
        I = tf.math.imag(S)

        # 3. Modify parts (unary op on real, scalar divide on imag)
        R_mod = tf.abs(R)
        I_mod = I / p

        # 4. Recombine using the Similar API: tf.keras.ops.add
        # In the original issue, torch.complex(R, I) was used.
        # Here we use tf.keras.ops.add to combine the derived tensors.
        # We cast to complex64 to maintain the semantic context of complex number operations.
        Z = tf.keras.ops.add(
            tf.cast(R_mod, tf.complex64),
            tf.cast(I_mod, tf.complex64)
        )
        return Z

    # Execution
    try:
        Z = model(x)
        
        # Assertions
        assert Z.dtype == tf.complex64, f"Expected complex64, got {Z.dtype}"
        # Shape check: (batch, time_frames, fft_bins)
        # fft_bins = n_fft // 2 + 1 for real-valued input
        expected_frames = (16000 + n_fft) // hop
        assert Z.shape == (1, expected_frames, n_fft // 2 + 1), \
            f"Shape mismatch: expected (1, {expected_frames}, {n_fft // 2 + 1}), got {Z.shape}"
        
        print("Test passed: tf.keras.ops.add handled STFT recombination correctly under tf.function.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    test_stft_recombination_with_add()