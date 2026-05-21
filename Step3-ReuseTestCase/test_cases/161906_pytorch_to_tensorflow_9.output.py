import torch
import tensorflow as tf
import numpy as np

def test_broadcast_arrays_stft_logic():
    """
    Adapted test case for tf.experimental.numpy.broadcast_arrays based on 
    the PyTorch torch.complex stride assertion bug.
    
    The original bug involved operations on real/imaginary parts (unary ops, 
    division by parameter) followed by a combination step (torch.complex).
    Here, we perform similar operations and use broadcast_arrays to align 
    the tensors, testing both eager and compiled (tf.function) execution.
    """
    
    # Parameters mirroring the PyTorch model
    n_fft = 512
    hop = 160
    win = 320
    
    # Input data
    # Using a fixed seed for reproducibility
    tf.random.set_seed(0)
    x = tf.random.normal((1, 16000), dtype=tf.float32)
    
    # Parameter mimicking nn.Parameter
    p = tf.Variable(2.0, dtype=tf.float32)

    # Logic to be tested
    def logic_fn(x_input, p_param):
        # 1. Compute STFT
        # tf.signal.stft returns a complex64 tensor
        S = tf.signal.stft(
            x_input, 
            frame_length=win, 
            frame_step=hop, 
            fft_length=n_fft,
            pad_end=True
        )
        
        # 2. Operations on Real and Imaginary parts
        # R = torch.abs(S.real)
        R = tf.abs(tf.math.real(S))
        
        # I = S.imag / self.p
        I = tf.math.imag(S) / p_param
        
        # 3. Target API: broadcast_arrays
        # In the original bug, torch.complex(R, I) failed due to stride issues.
        # Here we test if broadcast_arrays handles the resulting tensors correctly.
        R_b, I_b = tf.experimental.numpy.broadcast_arrays(R, I)
        
        return R_b, I_b

    # --- Eager Execution ---
    print("Testing Eager execution...")
    try:
        R_eager, I_eager = logic_fn(x, p)
        
        # Assertions
        assert R_eager.dtype == tf.float32
        assert I_eager.dtype == tf.float32
        assert R_eager.shape == I_eager.shape
        # Verify broadcasting didn't alter values where shapes matched
        assert tf.reduce_all(tf.equal(R_eager, tf.abs(tf.math.real(tf.signal.stft(x, win, hop, n_fft, pad_end=True)))))
        
        print(f"Eager mode OK: Shapes {R_eager.shape}, Broadcast successful")
    except Exception as e:
        print(f"Eager mode FAILED: {e}")

    # --- Compiled Execution (tf.function) ---
    print("\nTesting Compiled execution (tf.function)...")
    try:
        compiled_logic = tf.function(logic_fn)
        R_compiled, I_compiled = compiled_logic(x, p)
        
        # Assertions
        assert R_compiled.dtype == tf.float32
        assert I_compiled.dtype == tf.float32
        assert R_compiled.shape == I_eager.shape
        # Verify consistency with eager mode
        assert tf.reduce_all(tf.equal(R_compiled, R_eager))
        assert tf.reduce_all(tf.equal(I_compiled, I_eager))
        
        print(f"Compiled mode OK: Shapes {R_compiled.shape}, Broadcast successful")
    except Exception as e:
        print(f"Compiled mode FAILED: {e}")

if __name__ == "__main__":
    test_broadcast_arrays_stft_logic()