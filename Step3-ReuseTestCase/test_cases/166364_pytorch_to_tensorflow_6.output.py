import torch
import tensorflow as tf
import numpy as np

# Ensure we are in graph mode to properly test compat.v1 APIs
tf.compat.v1.disable_eager_execution()

def test_name_scope_with_learnable_scalar():
    """
    Adapted test case for tf.compat.v1.name_scope based on PyTorch Flex Attention bug.
    
    Original Bug Logic:
    1. Define a learnable scalar parameter.
    2. Use this scalar to modify a batched score tensor (score = score + scalar).
    3. Verify forward and backward passes.
    
    TensorFlow Adaptation:
    1. Define a learnable scalar variable (tf.Variable).
    2. Perform the addition operation inside a tf.compat.v1.name_scope context.
    3. Verify that the scalar broadcasts correctly over the batched tensor and gradients flow.
    """
    with tf.compat.v1.Session() as sess:
        # 1. Define a learnable scalar (mimicking nn.Parameter(torch.tensor(0.0)))
        # We use a variable initialized to 0.0 to match the bug report
        temp = tf.compat.v1.get_variable(
            "learnable_scalar", 
            shape=[], 
            dtype=tf.float32, 
            initializer=tf.zeros_initializer()
        )

        # 2. Define a batched input (mimicking the 'score' tensor in attention)
        # Shape: [Batch_Size, Head_Dim]
        batch_size = 2
        head_dim = 4
        score_input = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, head_dim])

        # 3. Define the logic inside the target API: tf.compat.v1.name_scope
        # This mimics the score_mod function: score = score + temp
        with tf.compat.v1.name_scope("score_modification"):
            # The operation adds the scalar to the batched tensor
            # In the original bug, this caused issues with vmap/compilation.
            # Here we verify it works within the name scope.
            modified_score = score_input + temp

        # 4. Run Forward Pass
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Create dummy input data
        input_data = np.random.randn(batch_size, head_dim).astype(np.float32)
        
        # Execute the graph
        result = sess.run(modified_score, feed_dict={score_input: input_data})
        
        # Verify broadcasting works (Scalar added to Batch)
        # Since temp is 0.0, result should equal input_data
        expected = input_data + 0.0
        assert np.allclose(result, expected), "Forward pass failed: Scalar did not broadcast correctly within name_scope."

        # 5. Run Backward Pass (Gradient Check)
        # The original bug failed in backward. We verify gradients flow through the scope.
        loss = tf.reduce_sum(modified_score)
        grads = tf.gradients(loss, temp)
        
        grad_val = sess.run(grads, feed_dict={score_input: input_data})
        
        # Gradient of sum(x + c) w.r.t c is the number of elements in x
        expected_grad = batch_size * head_dim
        assert np.allclose(grad_val[0], expected_grad), "Backward pass failed: Gradients incorrect for scalar variable."

        print("Test passed: tf.compat.v1.name_scope handles learnable scalars correctly.")

if __name__ == "__main__":
    test_name_scope_with_learnable_scalar()