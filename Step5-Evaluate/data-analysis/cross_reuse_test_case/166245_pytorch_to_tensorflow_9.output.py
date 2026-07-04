import torch
import tensorflow as tf
import numpy as np

# Replicate the seed and environment setup from the PyTorch bug report
tf.random.set_seed(751735337)

# Determine device (mimicking 'device=cuda' if available)
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

print(f"Running test on device: {device}")

with tf.device(device):
    # --- Replicate the tensor generation logic from the fuzzed program ---
    
    # arg_0: size=(15, 108, 4), dtype=int16
    # In PyTorch: var_node_4 = arg_0
    arg_0 = tf.random.uniform((15, 108, 4), minval=0, maxval=100, dtype=tf.int16)

    # var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0] -> size=(15, 27, 4)
    # 108 / 4 = 27
    var_node_3 = tf.split(arg_0, 4, axis=1)[0]

    # var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] -> size=(15, 27, 1)
    # 4 / 4 = 1
    var_node_2 = tf.split(var_node_3, 4, axis=2)[0]

    # var_node_1 = torch.squeeze(var_node_2) -> size=(15, 27)
    var_node_1 = tf.squeeze(var_node_2, axis=-1)

    # var_node_8 = torch.full((13, 27), 3, dtype=torch.int16)
    var_node_8 = tf.cast(tf.fill((13, 27), 3), tf.int16)

    # _input_size_var_node_7 = var_node_8.size(0) -> 13
    # _index_var_node_7 = torch.randint(0, 13, (11,))
    _index_var_node_7 = tf.random.uniform((11,), minval=0, maxval=13, dtype=tf.int32)

    # var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7) -> size=(11, 27)
    var_node_7 = tf.gather(var_node_8, _index_var_node_7, axis=0)

    # var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
    var_node_6 = tf.clip_by_value(var_node_7, clip_value_min=-1.0, clip_value_max=1.0)

    # arg_2: size=(3, 27), dtype=int16
    arg_2 = tf.random.uniform((3, 27), minval=0, maxval=100, dtype=tf.int16)

    # var_node_11 = torch.clamp(arg_2, min=-1.0, max=1.0)
    var_node_11 = tf.clip_by_value(arg_2, clip_value_min=-1.0, clip_value_max=1.0)

    # _input_size_var_node_10 = var_node_11.size(0) -> 3
    # _index_var_node_10 = torch.randint(0, 3, (1,))
    _index_var_node_10 = tf.random.uniform((1,), minval=0, maxval=3, dtype=tf.int32)

    # var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10) -> size=(1, 27)
    var_node_10 = tf.gather(var_node_11, _index_var_node_10, axis=0)

    # arg_3, arg_4, arg_5: size=(1, 27), dtype=int16
    arg_3 = tf.random.uniform((1, 27), minval=0, maxval=100, dtype=tf.int16)
    arg_4 = tf.random.uniform((1, 27), minval=0, maxval=100, dtype=tf.int16)
    arg_5 = tf.random.uniform((1, 27), minval=0, maxval=100, dtype=tf.int16)

    # var_node_15 = torch.cat([arg_3, arg_4, arg_5], dim=0) -> size=(3, 27)
    var_node_15 = tf.concat([arg_3, arg_4, arg_5], axis=0)

    # _input_size_var_node_14 = var_node_15.size(0) -> 3
    # _index_var_node_14 = torch.randint(0, 3, (1,))
    _index_var_node_14 = tf.random.uniform((1,), minval=0, maxval=3, dtype=tf.int32)

    # var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14) -> size=(1, 27)
    var_node_14 = tf.gather(var_node_15, _index_var_node_14, axis=0)

    # --- Test the Similar API: tf.keras.metrics.top_k_categorical_accuracy ---
    
    # The PyTorch code generates tensors of shape (N, 27). 
    # We will use var_node_1 (15, 27) to test the metric.
    # y_true expects one-hot or class indices. We use the generated int16 tensor.
    # y_pred expects probabilities (float). We cast the generated tensor to float.
    
    y_true = var_node_1
    y_pred = tf.cast(var_node_1, tf.float32)
    
    k = 5
    
    # Note: tf.keras.metrics.top_k_categorical_accuracy is a functional API in some versions
    # and a class in others. We use the functional form here as implied by the prompt.
    try:
        accuracy = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)
    except AttributeError:
        # Fallback for newer TF versions where it might be strictly a class or moved
        metric = tf.keras.metrics.TopKCategoricalAccuracy(k=k)
        metric.update_state(y_true, y_pred)
        accuracy = metric.result()

    print(f"Computed Top-K Accuracy shape: {accuracy.shape}")
    print(f"Computed Top-K Accuracy values: {accuracy.numpy()}")

    # Assertions to verify behavior
    # 1. Shape assertion: Should match batch size (15)
    assert accuracy.shape == (15,), f"Expected shape (15,), got {accuracy.shape}"
    
    # 2. Value assertion: 
    # Since y_true and y_pred contain the same values (just cast), argmax(y_true) == argmax(y_pred).
    # Therefore, the true class is always in the top K predictions.
    # Accuracy should be 1.0 for all samples.
    expected_accuracy = np.ones((15,), dtype=np.float32)
    np.testing.assert_array_almost_equal(accuracy.numpy(), expected_accuracy, decimal=5,
                                         err_msg="Expected accuracy to be 1.0 since y_true and y_pred are identical")

    print("Test passed successfully.")