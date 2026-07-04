import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility, mirroring the PyTorch fuzzer setup
tf.random.set_seed(751735337)

def test_cropping1d_divergence():
    """
    Adapted test case for tf.keras.layers.Cropping1D based on a PyTorch fuzzer report.
    The original report involved an eager/compile divergence with torch.gather/chunk.
    This test adapts the tensor manipulation logic to TensorFlow, using Cropping1D
    to mimic the dimension reduction behavior of torch.chunk.
    """
    
    # Define inputs mirroring the PyTorch shapes and dtypes
    # arg_0: size=(15, 108, 4), dtype=int16
    arg_0 = tf.cast(tf.random.uniform((15, 108, 4), minval=0, maxval=100, dtype=tf.int32), tf.int16)
    
    # arg_1: size=(11,), dtype=int64
    arg_1 = tf.random.uniform((11,), minval=0, maxval=100, dtype=tf.int64)
    
    # arg_2: size=(3, 27), dtype=int16
    arg_2 = tf.cast(tf.random.uniform((3, 27), minval=0, maxval=100, dtype=tf.int32), tf.int16)
    
    # arg_3, arg_4, arg_5: size=(1, 27), dtype=int16
    arg_3 = tf.cast(tf.random.uniform((1, 27), minval=0, maxval=100, dtype=tf.int32), tf.int16)
    arg_4 = tf.cast(tf.random.uniform((1, 27), minval=0, maxval=100, dtype=tf.int32), tf.int16)
    arg_5 = tf.cast(tf.random.uniform((1, 27), minval=0, maxval=100, dtype=tf.int32), tf.int16)
    
    # arg_6: size=(1,), dtype=int64
    arg_6 = tf.random.uniform((1,), minval=0, maxval=100, dtype=tf.int64)

    # Instantiate the similar API: Cropping1D
    # PyTorch used chunk(dim=1, 4) on size 108 -> 27. 
    # Cropping1D crops the time dimension (axis 1). 
    # To get 27 from 108, we crop (0, 81) to keep the first 27 elements.
    cropping_layer = tf.keras.layers.Cropping1D(cropping=(0, 81))

    # The core logic wrapped in tf.function to test for compilation divergence
    @tf.function
    def fuzzed_program_tf(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6):
        # --- Adaptation of Chunk/Squeeze logic using Cropping1D ---
        var_node_4 = arg_0 # (15, 108, 4)
        
        # Original: var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0] -> (15, 27, 4)
        # Adaptation: Use Cropping1D to reduce axis 1
        var_node_3 = cropping_layer(var_node_4)
        
        # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] -> (15, 27, 1)
        # Cropping1D only works on axis 1. We use slicing for axis 2 to mimic the shape change.
        var_node_2 = var_node_3[:, :, :1] 
        
        # Original: var_node_1 = torch.squeeze(var_node_2) -> (15, 27)
        var_node_1 = tf.squeeze(var_node_2, axis=-1)

        # --- Adaptation of Index/Clamp logic ---
        # Original: var_node_8 = torch.full((13, 27), 3, dtype=torch.int16)
        var_node_8 = tf.fill((13, 27), tf.cast(3, tf.int16))
        
        # Original: var_node_9 = arg_1
        var_node_9 = arg_1
        
        # Original: _input_size_var_node_7 = var_node_8.size(0)
        _input_size_var_node_7 = tf.shape(var_node_8)[0]
        
        # Original: _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), ...)
        # Note: maxval must be strictly greater than minval
        _index_var_node_7 = tf.random.uniform((11,), minval=0, maxval=tf.cast(_input_size_var_node_7, tf.int32), dtype=tf.int32)
        
        # Original: var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7)
        var_node_7 = tf.gather(var_node_8, _index_var_node_7, axis=0)
        
        # Original: var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
        var_node_6 = tf.clip_by_value(var_node_7, clip_value_min=-1.0, clip_value_max=1.0)
        
        # Original: var_node_12 = arg_2
        var_node_12 = arg_2
        
        # Original: var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0)
        var_node_11 = tf.clip_by_value(var
    assert size
