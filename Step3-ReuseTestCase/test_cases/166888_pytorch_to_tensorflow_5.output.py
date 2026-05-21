import torch
import tensorflow.compat.v1 as tf
import numpy as np

# Disable eager execution to use TF 1.x graph mode, which is required for 
# tf.compat.v1.train.QueueRunner and related APIs.
tf.disable_v2_behavior()

def test_queue_runner_with_scalar_clamp():
    """
    Adapts the PyTorch test case for torch.compile to TensorFlow's 
    tf.compat.v1.train.add_queue_runner.
    
    Original Logic:
        def f(x, max_val):
            y = torch.clamp(x, 0, max_val.item())
            return y
    """
    
    # 1. Define the graph inputs (mimicking function arguments)
    # x: A float tensor
    x = tf.placeholder(tf.float32, shape=[10, 20, 30], name='x')
    # max_val: A scalar tensor (mimicking torch.tensor(5.0))
    max_val = tf.placeholder(tf.float32, shape=[], name='max_val')

    # 2. Define the core logic (mimicking the function body)
    # PyTorch: y = torch.clamp(x, 0, max_val.item())
    # TensorFlow: tf.maximum(tf.minimum(x, max_val), 0.0)
    # Note: In TF graph mode, we use the tensor 'max_val' directly. 
    # The PyTorch bug was triggered by the .item() call during compilation.
    y = tf.maximum(tf.minimum(x, max_val), 0.0, name='y')

    # 3. Setup for the API under test: tf.compat.v1.train.add_queue_runner
    # We create a queue to enqueue the results of the operation.
    queue = tf.FIFOQueue(capacity=10, dtypes=[tf.float32], shapes=[x.get_shape()])
    enqueue_op = queue.enqueue(y)

    # Create a QueueRunner that will manage the enqueue operation
    qr = tf.train.QueueRunner(queue, [enqueue_op])

    # 4. Call the API (Similar to torch.compile)
    # This adds the QueueRunner to the graph collection.
    tf.compat.v1.train.add_queue_runner(qr)

    # Verify the runner was added
    runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    assert qr in runners, "QueueRunner was not added to the collection."

    # 5. Execution (Similar to compiled_func(x, max_val))
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        
        # Start the queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
        # Prepare input data
        x_data = np.random.randn(10, 20, 30).astype(np.float32)
        max_val_data = np.array(5.0, dtype=np.float32)
        
        # Run the enqueue operation manually to feed data into the queue
        # (Standard QueueRunners typically loop on ops that don't require external feeds,
        # but we trigger the logic here to verify the operation works).
        sess.run(enqueue_op, feed_dict={x: x_data, max_val: max_val_data})
        
        # Dequeue the result to verify the logic
        result = sess.run(queue.dequeue())
        
        # Verify the clamp logic (values should be between 0 and 5.0)
        expected = np.clip(x_data, 0, 5.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Cleanup
        coord.request_stop()
        coord.join(threads)

    print("Test passed: tf.compat.v1.train.add_queue_runner handled the logic correctly.")

if __name__ == "__main__":
    test_queue_runner_with_scalar_clamp()