import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_backend_variable_adapted():
    """
    Adapted test case for tf.keras.backend.variable based on PyTorch fuzzer output.
    The original PyTorch code involved tensor manipulations (chunk, squeeze, index_select, clamp, cat)
    leading to specific tensor shapes and dtypes. This test replicates that logic in TensorFlow
    and verifies that tf.keras.backend.variable can correctly instantiate variables from the results.
    """
    
    # Set seeds for reproducibility
    np.random.seed(751735337)
    tf.random.set_seed(751735337)

    # Initialize inputs (mimicking arg_0 to arg_6 from the PyTorch snippet)
    # PyTorch: size=(15, 108, 4), dtype=int16
    arg_0 = np.random.randint(0, 100, (15, 108, 4), dtype=np.int16)
    # PyTorch: size=(11,), dtype=int64
    arg_1 = np.random.randint(0, 100, (11,), dtype=np.int64)
    # PyTorch: size=(3, 27), dtype=int16
    arg_2 = np.random.randint(0, 100, (3, 27), dtype=np.int16)
    # PyTorch: size=(1, 27), dtype=int16
    arg_3 = np.random.randint(0, 100, (1, 27), dtype=np.int16)
    arg_4 = np.random.randint(0, 100, (1, 27), dtype=np.int16)
    arg_5 = np.random.randint(0, 100, (1, 27), dtype=np.int16)
    # PyTorch: size=(1,), dtype=int64
    arg_6 = np.random.randint(0, 100, (1,), dtype=np.int64)

    # --- Logic Translation ---

    # var_node_4 = arg_0
    var_node_4 = tf.convert_to_tensor(arg_0)

    # var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0]
    # 108 / 4 = 27, so splits are equal.
    var_node_3 = tf.split(var_node_4, 4, axis=1)[0]

    # var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    # 4 / 4 = 1, so splits are equal.
    var_node_2 = tf.split(var_node_3, 4, axis=2)[0]

    # var_node_1 = torch.squeeze(var_node_2)
    var_node_1 = tf.squeeze(var_node_2)

    # var_node_8 = torch.full((13, 27), 3, dtype=torch.int16)
    var_node_8 = tf.fill((13, 27), tf.cast(3, tf.int16))

    # var_node_9 = arg_1
    var_node_9 = tf.convert_to_tensor(arg_1)

    # _input_size_var_node_7 = var_node_8.size(0)
    _input_size_var_node_7 = tf.shape(var_node_8)[0]

    # _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), ...)
    # tf.random.uniform maxval is exclusive.
    _index_var_node_7 = tf.random.uniform((11,), minval=0, maxval=_input_size_var_node_7, dtype=tf.int64)

    # var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7)
    # Equivalent to tf.gather
    var_node_7 = tf.gather(var_node_8, _index_var_node_7, axis=0)

    # var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
    var_node_6 = tf.clip_by_value(var_node_7, clip_value_min=-1.0, clip_value_max=1.0)

    # var_node_12 = arg_2
    var_node_12 = tf.convert_to_tensor(arg_2)

    # var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0)
    var_node_11 = tf.clip_by_value(var_node_12, clip_value_min=-1.0, clip_value_max=1.0)

    # var_node_13 = torch.full((1,), 3, dtype=torch.int64)
    var_node_13 = tf.fill((1,), tf.cast(3, tf.int64))

    # _input_size_var_node_10 = var_node_11.size(0)
    _input_size_var_node_10 = tf.shape(var_node_11)[0]

    # _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), ...)
    _index_var_node_10 = tf.random.uniform((1,), minval=0, maxval=_input_size_var_node_10, dtype=tf.int64)

    # var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10)
    var_node_10 = tf.gather(var_node_11, _index_var_node_10, axis=0)

    # var_node_16 = arg_3, var_node_17 = arg_4, var_node_18 = arg_5
    var_node_16 = tf.convert_to_tensor(arg_3)
    var_node_17 = tf.convert_to_tensor(arg_4)
    var_node_18 = tf.convert_to_tensor(arg_5)

    # var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0)
    var_node_15 = tf.concat([var_node_16, var_node_17, var_node_18], axis=0)

    # var_node_20 = arg_6
    var_node_20 = tf.convert_to_tensor(arg_6)

    # var_node_19 = torch.clamp(var_node_20, min=None, max=1.0)
    # Equivalent to min(input, 1.0)
    var_node_19 = tf.minimum(var_node_20, tf.cast(1.0, tf.int64))

    # _input_size_var_node_14 = var_node_15.size(0)
    _input_size_var_node_14 = tf.shape(var_node_15)[0]

    # _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), ...)
    _index_var_node_14 = tf.random.uniform((1,), minval=0, maxval=_input_size_var_node_14, dtype=tf.int64)

    # var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14)
    var_node_14 = tf.gather(var_node_15, _index_var_node_14, axis=0)

    # var_node_22 = torch.full((4, 27), 3, dtype=torch.int16)
    var_node_22 = tf.fill((4, 27), tf.cast(3, tf.int16))

    # --- Testing tf.keras.backend.variable ---
    # We create Keras variables from the tensors generated above to verify the API behavior.
    
    # Test creating variable from squeezed tensor
    kvar_1 = tf.keras.backend.variable(var_node_1)
    assert isinstance(kvar_1, tf.Variable)
    assert kvar_1.shape == (15, 27)
    assert kvar_1.dtype == tf.int16

    # Test creating variable from clamped/gathered tensor
    kvar_6 = tf.keras.backend.variable(var_node_6)
    assert kvar_6.shape == (11, 27)
    assert kvar_6.dtype == tf.int16

    # Test creating variable from single-row gathered tensor
    kvar_10 = tf.keras.backend.variable(var_node_10)
    assert kvar_10.shape == (1, 27)
    assert kvar_10.dtype == tf.int16

    # Test creating variable from concatenated/gathered tensor
    kvar_14 = tf.keras.backend.variable(var_node_14)
    assert kvar_14.shape == (1, 27)
    assert kvar_14.dtype == tf.int16

    # Test creating variable from full tensor
    kvar_22 = tf.keras.backend.variable(var_node_22)
    assert kvar_22.shape == (4, 27)
    assert kvar_22.dtype == tf.int16

    print("Test passed: tf.keras.backend.variable handles adapted logic correctly.")

if __name__ == "__main__":
    test_tf_keras_backend_variable_adapted()