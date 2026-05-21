import torch
import tensorflow as tf
import numpy as np
import sys

def test_tf_keras_backend_prod():
    """
    Adapted test case for tf.keras.backend.prod based on the PyTorch conv1d bug report.
    The original bug highlights an eager/compile divergence with specific tensor strides,
    shapes, and mixed precision (bfloat16/float32).
    
    This test verifies if tf.keras.backend.prod exhibits similar behavior under
    tf.function (graph mode/compilation) compared to eager execution.
    """
    
    # Enable XLA compilation to mimic the aggressive compilation of torch._inductor
    # Note: This requires a compatible environment (GPU/TPU or proper CPU setup)
    # If XLA is not available, tf.function will still run in graph mode.
    try:
        # Check if GPU is available to match the 'cuda' device in the original bug
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    # Mimic the tensor setup from the PyTorch bug
    # Original: t4 size=(17, 64, 358), dtype=float32
    # Original: t2 size=(17, 261, 358), dtype=bfloat16 (transposed)
    
    # We create a tensor with shape (17, 64, 358) similar to t4
    # We start with bfloat16 to mimic the mixed precision context (t0 -> t1 -> t2)
    # then cast to float32 before the operation, similar to t3 -> t4.
    
    # Input tensor
    # Shape (17, 64, 358)
    arg0 = tf.random.normal([17, 64, 358], dtype=tf.bfloat16)
    
    # Cast to float32
    arg0_float = tf.cast(arg0, tf.float32)
    
    # Perform a transpose to mimic the non-contiguous stride issue in the bug
    # PyTorch: t2 = t1.transpose(1, 0) resulting in stride=(93438, 358, 1)
    # Here we transpose (17, 64, 358) -> (64, 17, 358)
    input_tensor = tf.transpose(arg0_float, [1, 0, 2])

    # The function to test
    def foo(x):
        # Original API: torch.nn.functional.conv1d
        # Target API: tf.keras.backend.prod
        # We apply prod along axis 1 (which corresponds to the 17 dimension after transpose)
        # This is a reduction operation, unlike conv1d, but we test the compilation stability.
        return tf.keras.backend.prod(x, axis=1)

    # 1. Eager Execution
    print("Running Eager Execution...")
    try:
        out_eager = foo(input_tensor)
        print(f"Eager Success!  Output shape: {out_eager.shape}")
    except Exception as e:
        print(f"Eager Failed!  Error: {e}")
        return

    # 2. Compiled Execution (tf.function)
    print("Running Compiled Execution (tf.function)...")
    try:
        # Using jit_compile=True to stress the compiler like torch.compile
        compiled_foo = tf.function(foo, jit_compile=True)
        out_compiled = compiled_foo(input_tensor)
        print(f"Compile Success!  Output shape: {out_compiled.shape}")
    except Exception as e:
        print(f"Compile Failed!  Error: {e}")
        # In the context of the original bug, this is where the failure occurred
        return

    # 3. Verification
    # Check for numerical divergence
    if out_eager.shape != out_compiled.shape:
        print(f"Shape Divergence Detected! Eager: {out_eager.shape}, Compiled: {out_compiled.shape}")
    else:
        # Use allclose for float comparison
        if np.allclose(out_eager.numpy(), out_compiled.numpy()):
            print("Verification Passed: Eager and Compiled outputs match. ")
        else:
            print("Verification Failed: Eager and Compiled outputs diverge. ")
            diff = np.abs(out_eager.numpy() - out_compiled.numpy())
            print(f"Max difference: {np.max(diff)}")

if __name__ == '__main__':
    test_tf_keras_backend_prod()