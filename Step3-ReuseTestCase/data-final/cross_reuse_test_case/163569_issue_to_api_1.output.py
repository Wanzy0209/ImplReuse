import torch
import tensorflow as tf
import numpy as np

def test_tf_conv3d_transpose_divergence():
    """
    Test case adapted from PyTorch Issue 163569.
    Original Issue: torch.nn.functional.conv1d works in eager but fails compile.
    Similar API: tf.compat.v1.nn.conv3d_transpose.
    
    This test checks for eager vs graph (tf.function) divergence in TensorFlow
    using a logic pattern similar to the PyTorch bug report, adapted for 3D convolution.
    """
    
    # Enable/disable v2 behavior to ensure compat.v1 works as expected
    tf.compat.v1.disable_eager_execution() 
    # Note: We actually want to test eager vs graph, so we will run the function 
    # in eager mode first, then wrap it. 
    # However, disable_eager_execution() globally disables eager mode. 
    # To properly test both, we should NOT disable it globally, but rather 
    # rely on tf.function for the graph part.
    # Reverting the global disable to allow the first eager run.
    tf.compat.v1.enable_eager_execution()

    # Define inputs mirroring the PyTorch shapes, expanded for 3D (NCHW format)
    # PyTorch arg0: (2, 261, 17, 358) -> TF: (Batch, Channel, Depth, Height, Width)
    # We add a Channel dimension (1) to arg0 to make it 5D.
    arg0 = tf.random.normal([2, 1, 261, 17, 358], dtype=tf.float32)
    
    # PyTorch arg1: (17, 64, 358) -> TF: (17, 64, 1, 1, 358)
    # Batch=17, Channels=64, Depth=1, Height=1, Width=358
    arg1 = tf.random.normal([17, 64, 1, 1, 358], dtype=tf.float32)
    
    # PyTorch arg2 (weight): (261, 1, 64) -> Transposed -> (261, 64, 1)
    # TF Filter for conv3d_transpose: [Depth, Height, Width, OutChannels, InChannels]
    # PyTorch conv1d weight: (OutChannels, InChannels, KernelSize) -> (261, 64, 1)
    # TF Filter: [1, 1, 1, 261, 64]
    arg2 = tf.random.normal([1, 1, 1, 261, 64], dtype=tf.float32)

    def logic_fn(a0, a1, a2):
        # t0 = arg0
        t0 = a0
        
        # t1 = t0.max(dim=0).values
        # Reduces batch dimension (dim 0)
        t1 = tf.reduce_max(t0, axis=0) # Shape: (1, 261, 17, 358)
        
        # t2 = t1.transpose(1, 0)
        # PyTorch: (261, 17, 358) -> (17, 261, 358)
        # TF (NCHW): (1, 261, 17, 358) -> Transpose Depth and Height (indices 1 and 2)
        # Result: (1, 17, 261, 358)
        t2 = tf.transpose(t1, [0, 2, 1, 3])
        
        # t3 = arg1
        t3 = a1
        
        # t4 = torch.exp(t3)
        t4 = tf.exp(t3)
        
        # t7 = torch.nn.functional.conv1d(t4, t6, stride=1, padding=0)
        # TF equivalent: conv3d_transpose
        # Input t4: (17, 64, 1, 1, 358)
        # Filter a2: (1, 1, 1, 261, 64)
        # Output Shape: (17, 261, 1, 1, 358)
        output_shape = tf.constant([17, 261, 1, 1, 358])
        strides = [1, 1, 1, 1, 1]
        padding = 'VALID' # Corresponds to padding=0 in PyTorch
        
        t7 = tf.compat.v1.nn.conv3d_transpose(
            t4, 
            a2, 
            output_shape=output_shape, 
            strides=strides, 
            padding=padding,
            data_format='NCHW' 
        )
        
        # t8 = t7.clone(); t8.zero_()
        t8 = tf.identity(t7)
        t8 = tf.zeros_like(t8)
        
        # t9 = t2 * t7 * t8
        # t2: (1, 17, 261, 358). 
        # t7: (17, 261, 1, 1, 358).
        # To multiply, we reshape t2 to match t7's dimensions (Batch, Channel, Depth, Height, Width).
        # PyTorch t2 was (17, 261, 358). PyTorch t7 was (17, 261, 358).
        # We reshape t2 to (17, 261, 1, 1, 358).
        t2_reshaped = tf.reshape(t2, [17, 261, 1, 1, 358])
        
        t9 = t2_reshaped * t7 * t8
        return t9

    # 1. Run in Eager Mode
    out_eager = logic_fn(arg0, arg1, arg2)
    
    # 2. Run in Graph Mode (tf.function)
    # This mimics the 'torch.compile' aspect of the original bug
    compiled_fn = tf.function(logic_fn)
    out_graph = compiled_fn(arg0, arg1, arg2)
    
    # 3. Check for divergence
    # The original bug reported a failure in compile mode. 
    # Here we assert that the outputs are close to ensure no divergence.
    try:
        np.testing.assert_allclose(out_eager.numpy(), out_graph.numpy(), rtol=1e-5, atol=1e-5)
        print("Test Passed: No divergence between Eager and Graph execution for tf.compat.v1.nn.conv3d_transpose.")
    except AssertionError as e:
        print(f"Test Failed: Divergence detected.\n{e}")
        raise

if __name__ == '__main__':
    test_tf_conv3d_transpose_divergence()