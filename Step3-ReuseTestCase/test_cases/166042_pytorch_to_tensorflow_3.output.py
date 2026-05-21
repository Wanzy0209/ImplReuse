import torch
import tensorflow as tf
import numpy as np

# Adapted from PyTorch fuzzer output to TensorFlow
# The original bug involves passing a bfloat16 tensor (result of matmul ops)
# to an API expecting specific types (int for indices in embedding).
# Here, we adapt this to tf.compat.v1.train.maybe_batch by passing the 
# generated bfloat16 tensor as the 'keep_input' argument, which expects a boolean tensor.

def test_maybe_batch_type_mismatch():
    # Set seed for reproducibility
    tf.random.set_seed(1352030645)
    np.random.seed(1352030645)

    # Disable eager execution to test graph compilation behavior, 
    # as the original bug mentions "Eager/Compile Divergence".
    # Note: tf.compat.v1.train.maybe_batch is a legacy queue-based op often used in graphs.
    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        # Define inputs (arg_0 to arg_5) matching the shapes in the PyTorch snippet
        # Using random data for arguments, specific constants for full() ops
        arg_0 = tf.constant(np.random.randn(4, 8), dtype=tf.bfloat16) # size=(4, 8)
        arg_1 = tf.constant(np.random.randn(7, 12), dtype=tf.bfloat16) # size=(7, 12)
        arg_2 = tf.constant(np.random.randn(12, 2), dtype=tf.bfloat16) # size=(12, 2)
        arg_3 = tf.constant(np.random.randn(14, 9), dtype=tf.bfloat16) # size=(14, 9)
        arg_4 = tf.constant(np.random.randn(2), dtype=tf.bfloat16) # size=(2,)
        arg_5 = tf.constant(np.random.randn(478, 13), dtype=tf.bfloat16) # size=(478, 13)

        # Replicate the computation graph from the PyTorch snippet
        # var_node_5 = torch.full((8, 7), -0.80078125, dtype=torch.bfloat16)
        var_node_5 = tf.fill([8, 7], tf.cast(-0.80078125, tf.bfloat16))
        
        # var_node_3 = torch.matmul(var_node_4, var_node_5)
        var_node_3 = tf.matmul(arg_0, var_node_5)
        
        # var_node_6 = torch.matmul(var_node_7, var_node_8)
        var_node_6 = tf.matmul(arg_1, arg_2)
        
        # var_node_2 = torch.matmul(var_node_3, var_node_6)
        var_node_2 = tf.matmul(var_node_3, var_node_6)
        
        # var_node_11 = torch.full((2, 3), 1.515625, dtype=torch.bfloat16)
        var_node_11 = tf.fill([2, 3], tf.cast(1.515625, tf.bfloat16))
        
        # var_node_12 = torch.full((3, 16), 0.2353515625, dtype=torch.bfloat16)
        var_node_12 = tf.fill([3, 16], tf.cast(0.2353515625, tf.bfloat16))
        
        # var_node_10 = torch.matmul(var_node_11, var_node_12)
        var_node_10 = tf.matmul(var_node_11, var_node_12)
        
        # var_node_14 = torch.full((16, 4), 2.21875, dtype=torch.bfloat16)
        var_node_14 = tf.fill([16, 4], tf.cast(2.21875, tf.bfloat16))
        
        # var_node_15 = torch.full((4, 9), -1.7421875, dtype=torch.bfloat16)
        var_node_15 = tf.fill([4, 9], tf.cast(-1.7421875, tf.bfloat16))
        
        # var_node_13 = torch.matmul(var_node_14, var_node_15)
        var_node_13 = tf.matmul(var_node_14, var_node_15)
        
        # var_node_9 = torch.matmul(var_node_10, var_node_13)
        var_node_9 = tf.matmul(var_node_10, var_node_13)
        
        # var_node_1 = torch.matmul(var_node_2, var_node_9)
        var_node_1 = tf.matmul(var_node_2, var_node_9)
        
        # var_node_20 = torch.full((9, 2), 0.8203125, dtype=torch.bfloat16)
        var_node_20 = tf.fill([9, 2], tf.cast(0.8203125, tf.bfloat16))
        
        # var_node_18 = torch.matmul(var_node_19, var_node_20)
        var_node_18 = tf.matmul(arg_3, var_node_20)
        
        # var_node_23 = torch.full((2,), -0.7421875, dtype=torch.bfloat16)
        var_node_23 = tf.fill([2], tf.cast(-0.7421875, tf.bfloat16))
        
        # var_node_21 = torch.add(var_node_22, var_node_23)
        var_node_21 = tf.add(arg_4, var_node_23)
        
        # var_node_17 = torch.matmul(var_node_18, var_node_21)
        # Resulting shape is (14,), dtype is bfloat16
        var_node_17 = tf.matmul(var_node_18, var_node_21)

        # The PyTorch bug involves passing a tensor of the wrong dtype (bfloat16 instead of int)
        # to an API expecting specific types (indices).
        # For tf.compat.v1.train.maybe_batch, 'keep_input' expects a bool Tensor.
        # We pass var_node_17 (bfloat16) here to test type handling/divergence.
        
        # Create a dummy tensor list to batch
        dummy_tensor = tf.constant([1.0], dtype=tf.float32)
        
        try:
            # This call attempts to use a bfloat16 tensor as a boolean control signal
            batched = tf.compat.v1.train.maybe_batch(
                tensors=[dummy_tensor],
                keep_input=var_node_17, # Type mismatch: bfloat16 vs expected bool
                batch_size=1,
                capacity=32
            )
            
            # Initialize variables and queue runners (required for legacy queue ops)
            init_op = tf.compat.v1.initialize_all_variables()
            sess.run(init_op)
            
            # Attempt to run the op
            # In the original bug, this might cause an assertion or divergence
            result = sess.run(batched)
            print("Test passed (or error was silently handled):", result)
            
        except Exception as e:
            # Catching the error to demonstrate the failure mode similar to the assert in the bug report
            print(f"Caught expected exception due to type mismatch: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_maybe_batch_type_mismatch()