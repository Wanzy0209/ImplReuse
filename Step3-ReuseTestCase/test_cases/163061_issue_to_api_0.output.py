import torch
import tensorflow as tf
import threading
import time
import numpy as np

def test_proximal_adagrad_gil_release():
    """
    Test case to verify if tf.compat.v1.train.ProximalAdagradOptimizer 
    releases the Global Interpreter Lock (GIL) during its execution, 
    similar to the issue reported for torch.compile.
    """
    # Ensure TF 1.x behavior for the compat API
    tf.compat.v1.disable_eager_execution()

    # 1. Setup the computational graph
    # Using large tensors to simulate the heavy kernel execution from the PyTorch issue
    dim = 4096
    x = tf.compat.v1.placeholder(tf.float32, shape=[dim, dim])
    y = tf.compat.v1.Variable(tf.random.normal([dim, dim]), name='weights')

    # Define a loss function (MSE)
    loss = tf.reduce_mean(tf.square(x - y))

    # Instantiate the Similar API: ProximalAdagradOptimizer
    optimizer = tf.compat.v1.train.ProximalAdagradOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    # 2. Logic to detect GIL holding
    # If the GIL is held by the main thread running the optimizer, 
    # a background thread attempting to run Python code will be blocked.
    gil_released_count = 0
    stop_event = threading.Event()

    def background_python_task():
        """
        A simple CPU-bound Python loop. 
        If this runs while the main thread is executing the TF op, 
        the GIL was released.
        """
        nonlocal gil_released_count
        while not stop_event.is_set():
            # Perform a trivial Python operation
            gil_released_count += 1
            # Sleep briefly to prevent 100% CPU usage in the thread itself
            time.sleep(0.001) 

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Create dummy data
        data = np.random.randn(dim, dim).astype(np.float32)

        # Start the background thread
        bg_thread = threading.Thread(target=background_python_task)
        bg_thread.start()

        # 3. Reproduce the execution pattern from the original issue
        # Original: for _ in range(10): torch_compile_add(x, y)
        # Adapted: Run the optimizer step multiple times
        iterations = 10
        start_time = time.time()
        
        for _ in range(iterations):
            sess.run(train_op, feed_dict={x: data})
            
        duration = time.time() - start_time

        # Signal the background thread to stop
        stop_event.set()
        bg_thread.join()

        # 4. Analysis
        # If the background thread managed to increment the counter significantly 
        # during the optimizer execution, it implies the GIL was released.
        print(f"Execution time for {iterations} iterations: {duration:.4f}s")
        print(f"Background thread iterations (GIL release indicator): {gil_released_count}")

        # We assert that the background thread was able to run.
        # If GIL was held (like the bug in torch.compile), this count would be very low (near 0).
        # Note: 100 is an arbitrary threshold to ensure the thread actually ran concurrently.
        assert gil_released_count > 100, \
            f"GIL appears to be held (background count: {gil_released_count}). " \
            "Expected GIL to be released during ProximalAdagradOptimizer execution."

if __name__ == "__main__":
    test_proximal_adagrad_gil_release()