import torch
import sys

# Attempt to import TensorFlow, handle environment errors gracefully
try:
    import tensorflow as tf
except ImportError as e:
    print("Skipping test: TensorFlow import failed.")
    print("This is likely due to a system environment issue (e.g., libstdc++ version mismatch).")
    print(f"Error details: {e}")
    sys.exit(0)

def test_argmax_bfloat16_compilation():
    """
    Test case for tf.math.argmax based on the PyTorch bug report (Issue 164063).
    
    The original bug involves a TypeError('unexpected type fp32') when performing
    a reduction operation (torch.var) on a bfloat16 tensor under eager/compile 
    divergence with specific precision emulation settings.
    
    This test adapts the logic to TensorFlow, verifying that tf.math.argmax
    (a similar reduction API) handles bfloat16 inputs correctly under XLA
    compilation (tf.function with jit_compile=True), ensuring no type casting
    errors occur.
    """
    
    # Enable XLA compilation to mimic torch.compile behavior
    @tf.function(jit_compile=True)
    def compiled_argmax_op(input_tensor):
        # Perform the reduction operation along a specific dimension.
        # In the original bug, t1.var(dim=2) was used.
        # Here we use tf.math.argmax along axis 2.
        return tf.math.argmax(input_tensor, axis=2)

    # Create a bfloat16 tensor matching the dimensions of the problematic tensor in the bug report.
    # Original t1 shape: (28, 24, 3, 127), dtype=bfloat16
    # We use CPU/CUDA based on availability, similar to the original 'cuda' device context.
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    
    with tf.device(device):
        # Initialize tensor with random values to ensure dynamic execution path
        t1 = tf.random.uniform((28, 24, 3, 127), minval=-10.0, maxval=10.0, dtype=tf.bfloat16)
        
        try:
            # Run the compiled operation
            result = compiled_argmax_op(t1)
            
            # Verify output shape (reduction on dim 2 removes that dimension)
            expected_shape = (28, 24, 127)
            assert result.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {result.shape}"
            
            # Verify output dtype (argmax typically returns int64)
            assert result.dtype == tf.int64, f"Dtype mismatch: expected int64, got {result.dtype}"
            
            print(f"TensorFlow Compilation Success on {device}! ")
            print(f"Input dtype: {t1.dtype}, Output dtype: {result.dtype}")
            return True
            
        except TypeError as e:
            # Catching the specific error type mentioned in the original bug
            print(f"TypeError encountered during compilation/execution: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

if __name__ == '__main__':
    test_argmax_bfloat16_compilation()