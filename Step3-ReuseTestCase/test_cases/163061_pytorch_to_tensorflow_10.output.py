import torch
import tensorflow as tf
import threading
import time

# Ensure compatibility mode for the specific API requested
tf.compat.v1.disable_eager_execution()

def test_range_input_producer_gil():
    """
    Test case to verify if tf.compat.v1.train.range_input_producer 
    releases the Global Interpreter Lock (GIL) during execution.
    
    This adapts the logic from the PyTorch bug report (Issue 163061) where 
    torch.compile was found to hold the GIL, preventing multi-threading 
    performance benefits.
    """
    
    # Shared state to monitor background thread activity
    shared_state = {'counter': 0, 'stop': False}

    def background_worker():
        """
        A thread that increments a counter. 
        If the GIL is released by the main thread, this thread will run 
        and increment the counter. If the GIL is held, this thread will 
        be blocked.
        """
        while not shared_state['stop']:
            shared_state['counter'] += 1
            # Sleep briefly to yield control, but the increment requires the GIL
            time.sleep(0.0001)

    # Start the background thread
    worker_thread = threading.Thread(target=background_worker)
    worker_thread.start()

    # --- Setup and Execution of the Similar API ---
    limit = 10000
    # Create the range input producer
    # This API is part of the tf.compat.v1 train input pipeline
    producer = tf.compat.v1.train.range_input_producer(
        limit, 
        num_epochs=None, 
        shuffle=False, 
        seed=None, 
        capacity=32, 
        shared_name=None, 
        name=None
    )
    
    # Dequeue a batch to actually trigger the queue operations
    batch_size = 100
    batch = producer.dequeue_many(batch_size)

    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for range_input_producer internal counters)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Run the operation multiple times to simulate a workload
        # and provide a window to check GIL status
        for _ in range(100):
            sess.run(batch)
    # ----------------------------------------------

    # Signal the background thread to stop
    shared_state['stop'] = True
    worker_thread.join()

    print(f"Background thread counter value: {shared_state['counter']}")

    # Assertion: If the GIL was released, the background thread should have 
    # been able to increment the counter significantly.
    # If the GIL was held (similar to the torch.compile bug), the counter 
    # would remain low (likely close to 0).
    assert shared_state['counter'] > 100, \
        "GIL appears to be held (counter did not increment). This mimics the torch.compile bug."

if __name__ == "__main__":
    test_range_input_producer_gil()