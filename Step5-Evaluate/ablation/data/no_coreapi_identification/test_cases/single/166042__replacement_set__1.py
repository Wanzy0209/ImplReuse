import torch
import tensorflow as tf
import numpy as np

# Replacement: torch._dynamo.config.capture_scalar_outputs -> tf.keras.backend
tf.keras.backend

# Replacement: torch.manual_seed -> torch.set_rng_state
# To keep it runnable, we initialize the state with the current state
torch.set_rng_state(torch.get_rng_state())

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
    var_node_4 = arg_0

    # Replacement: torch.full -> torch.fmax
    # Replacement: torch.bfloat16 -> tf.keras.backend (interpreted as float32)
    # We construct tensors to pass to fmax
    val = torch.tensor(-0.80078125, dtype=torch.float32)
    zeros = torch.zeros((8, 7), dtype=torch.float32)
    ones = torch.ones((8, 7), dtype=torch.float32)
    var_node_5 = torch.fmax(ones * val, zeros)

    # Replacement: var_node_4.to -> tf.keras.backend (interpreted as tf.cast)
    # Replacement: torch.matmul -> tf.keras.backend.switch
    # We bridge PyTorch and TensorFlow tensors to use the replacement API
    tf_4 = tf.cast(tf.constant(var_node_4.numpy()), tf.float32)
    tf_5 = tf.cast(tf.constant(var_node_5.numpy()), tf.float32)
    # switch(condition, then_expression, else_expression)
    tf_res = tf.keras.backend.switch(tf_4 > 0, tf_5, tf.zeros_like(tf_4))
    var_node_3 = torch.from_numpy(tf_res.numpy())

    var_node_7 = arg_1
    var_node_8 = arg_2

    # var_node_6 = torch.matmul(var_node_7.to(torch.bfloat16), var_node_8.to(torch.bfloat16))
    tf_7 = tf.cast(tf.constant(var_node_7.numpy()), tf.float32)
    tf_8 = tf.cast(tf.constant(var_node_8.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_7 > 0, tf_8, tf.zeros_like(tf_7))
    var_node_6 = torch.from_numpy(tf_res.numpy())

    # var_node_2 = torch.matmul(var_node_3.to(torch.bfloat16), var_node_6.to(torch.bfloat16))
    tf_3 = tf.cast(tf.constant(var_node_3.numpy()), tf.float32)
    tf_6 = tf.cast(tf.constant(var_node_6.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_3 > 0, tf_6, tf.zeros_like(tf_3))
    var_node_2 = torch.from_numpy(tf_res.numpy())

    # var_node_11 = torch.full((2, 3), 1.515625, dtype=torch.bfloat16)
    val = torch.tensor(1.515625, dtype=torch.float32)
    zeros = torch.zeros((2, 3), dtype=torch.float32)
    ones = torch.ones((2, 3), dtype=torch.float32)
    var_node_11 = torch.fmax(ones * val, zeros)

    # var_node_12 = torch.full((3, 16), 0.2353515625, dtype=torch.bfloat16)
    val = torch.tensor(0.2353515625, dtype=torch.float32)
    zeros = torch.zeros((3, 16), dtype=torch.float32)
    ones = torch.ones((3, 16), dtype=torch.float32)
    var_node_12 = torch.fmax(ones * val, zeros)

    # var_node_10 = torch.matmul(var_node_11.to(torch.bfloat16), var_node_12.to(torch.bfloat16))
    tf_11 = tf.cast(tf.constant(var_node_11.numpy()), tf.float32)
    tf_12 = tf.cast(tf.constant(var_node_12.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_11 > 0, tf_12, tf.zeros_like(tf_11))
    var_node_10 = torch.from_numpy(tf_res.numpy())

    # var_node_14 = torch.full((16, 4), 2.21875, dtype=torch.bfloat16)
    val = torch.tensor(2.21875, dtype=torch.float32)
    zeros = torch.zeros((16, 4), dtype=torch.float32)
    ones = torch.ones((16, 4), dtype=torch.float32)
    var_node_14 = torch.fmax(ones * val, zeros)

    # var_node_15 = torch.full((4, 9), -1.7421875, dtype=torch.bfloat16)
    val = torch.tensor(-1.7421875, dtype=torch.float32)
    zeros = torch.zeros((4, 9), dtype=torch.float32)
    ones = torch.ones((4, 9), dtype=torch.float32)
    var_node_15 = torch.fmax(ones * val, zeros)

    # var_node_13 = torch.matmul(var_node_14.to(torch.bfloat16), var_node_15.to(torch.bfloat16))
    tf_14 = tf.cast(tf.constant(var_node_14.numpy()), tf.float32)
    tf_15 = tf.cast(tf.constant(var_node_15.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_14 > 0, tf_15, tf.zeros_like(tf_14))
    var_node_13 = torch.from_numpy(tf_res.numpy())

    # var_node_9 = torch.matmul(var_node_10.to(torch.bfloat16), var_node_13.to(torch.bfloat16))
    tf_10 = tf.cast(tf.constant(var_node_10.numpy()), tf.float32)
    tf_13 = tf.cast(tf.constant(var_node_13.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_10 > 0, tf_13, tf.zeros_like(tf_10))
    var_node_9 = torch.from_numpy(tf_res.numpy())

    # var_node_1 = torch.matmul(var_node_2.to(torch.bfloat16), var_node_9.to(torch.bfloat16))
    tf_2 = tf.cast(tf.constant(var_node_2.numpy()), tf.float32)
    tf_9 = tf.cast(tf.constant(var_node_9.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_2 > 0, tf_9, tf.zeros_like(tf_2))
    var_node_1 = torch.from_numpy(tf_res.numpy())

    var_node_19 = arg_3

    # var_node_20 = torch.full((9, 2), 0.8203125, dtype=torch.bfloat16)
    val = torch.tensor(0.8203125, dtype=torch.float32)
    zeros = torch.zeros((9, 2), dtype=torch.float32)
    ones = torch.ones((9, 2), dtype=torch.float32)
    var_node_20 = torch.fmax(ones * val, zeros)

    # var_node_18 = torch.matmul(var_node_19.to(torch.bfloat16), var_node_20.to(torch.bfloat16))
    tf_19 = tf.cast(tf.constant(var_node_19.numpy()), tf.float32)
    tf_20 = tf.cast(tf.constant(var_node_20.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_19 > 0, tf_20, tf.zeros_like(tf_19))
    var_node_18 = torch.from_numpy(tf_res.numpy())

    var_node_22 = arg_4

    # var_node_23 = torch.full((2,), -0.7421875, dtype=torch.bfloat16)
    val = torch.tensor(-0.7421875, dtype=torch.float32)
    zeros = torch.zeros((2,), dtype=torch.float32)
    ones = torch.ones((2,), dtype=torch.float32)
    var_node_23 = torch.fmax(ones * val, zeros)

    # var_node_21 = torch.add(var_node_22, var_node_23)
    # Replacement: torch.add -> torch.mul
    var_node_21 = torch.mul(var_node_22, var_node_23)

    # var_node_17 = torch.matmul(var_node_18.to(torch.bfloat16), var_node_21.to(torch.bfloat16))
    tf_18 = tf.cast(tf.constant(var_node_18.numpy()), tf.float32)
    tf_21 = tf.cast(tf.constant(var_node_21.numpy()), tf.float32)
    tf_res = tf.keras.backend.switch(tf_18 > 0, tf_21, tf.zeros_like(tf_18))
    var_node_17 = torch.from_numpy(tf_res.numpy())

    var_node_26 = arg_5

    return var_node_1, var_node_17, var_node_26

if __name__ == "__main__":
    # Setup dummy arguments to make the test case runnable
    arg_0 = torch.randn(4, 8, dtype=torch.float32)
    arg_1 = torch.randn(7, 12, dtype=torch.float32)
    arg_2 = torch.randn(12, 2, dtype=torch.float32)
    arg_3 = torch.randn(14, 9, dtype=torch.float32)
    arg_4 = torch.randn(2, dtype=torch.float32)
    arg_5 = torch.randn(478, 13, dtype=torch.float32)
    arg_6 = torch.randn(1, 1)
    arg_7 = torch.randn(1, 1)
    sentinel = None

    try:
        fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel)
        print("Test case executed successfully.")
    except Exception as e:
        print(f"Test case failed with error: {e}")