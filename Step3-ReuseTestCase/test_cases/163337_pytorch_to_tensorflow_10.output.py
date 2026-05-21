import torch
import tensorflow as tf
import numpy as np

def test_gru_cell_fp16_rocm():
    """
    Adapted test case for tf.keras.layers.GRUCell based on the PyTorch 
    cpp_extension compilation issue (Issue 163337).

    Original Issue: Compilation error with static_cast from float to __half 
    when building a PyTorch extension with rocWMMA on ROCm.

    Adaptation Logic:
    The original bug occurs during the compilation of a custom C++ extension 
    specifically handling half-precision (fp16) data types on AMD GPUs.
    Since tf.keras.layers.GRUCell is a high-level API that does not support 
    compiling custom C++ sources on the fly, we cannot reproduce the exact 
    compilation error.

    However, to verify the similar API's behavior in the same context (ROCm/GPU 
    with fp16), we instantiate the GRUCell with float16 dtype and execute a 
    forward pass. This ensures that the underlying TensorFlow implementation 
    correctly handles fp16 operations on the GPU without the type conversion 
    errors seen in the custom PyTorch extension.
    """
    
    # Check for GPU availability (ROCm or CUDA)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPU available.")
        return

    print(f"Testing on GPU: {gpus[0].name}")

    # Parameters
    batch_size = 4
    input_size = 16
    units = 32
    
    # Use float16 to align with the __half / hfloat16_t context of the original bug
    dtype = tf.float16

    # Instantiate the GRUCell
    # Note: Unlike torch.utils.cpp_extension.load, this does not compile code.
    # It initializes a pre-compiled layer.
    gru_cell = tf.keras.layers.GRUCell(
        units=units, 
        dtype=dtype,
        # Reset_after=True is default for TF GRU, similar to standard CuDNN impls
        reset_after=True 
    )

    # Create dummy input data
    inputs = tf.random.normal([batch_size, input_size], dtype=dtype)
    
    # Initialize hidden state
    states = [tf.zeros([batch_size, units], dtype=dtype)]

    # Execute the layer (Forward pass)
    # This triggers the underlying GPU kernels
    output, new_states = gru_cell(inputs, states)

    # Assertions to verify correct behavior
    assert output.dtype == dtype, f"Expected dtype {dtype}, but got {output.dtype}"
    assert output.shape == (batch_size, units), f"Expected shape ({batch_size}, {units}), but got {output.shape}"
    assert new_states[0].shape == (batch_size, units), "Hidden state shape mismatch"
    
    print("Test passed: tf.keras.layers.GRUCell executed successfully with float16 on GPU.")

if __name__ == "__main__":
    test_gru_cell_fp16_rocm()