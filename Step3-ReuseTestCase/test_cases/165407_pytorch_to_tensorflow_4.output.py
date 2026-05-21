import torch
import tensorflow as tf
import tracemalloc
import gc

# Ensure TF 1.x behavior for the specific API
tf.compat.v1.disable_eager_execution()

def test_string_input_producer_leak():
    """
    Adapted test case for tf.compat.v1.train.string_input_producer.
    
    Original Issue: Memory leak in torch.compile where tensor counts increase 
    every step (ghost tensors).
    
    Adaptation Logic: 
    1. Use string_input_producer to create a data pipeline.
    2. Run a loop consuming data from the queue.
    3. Monitor memory allocation (Alloc/Res) to detect leaks, 
       as TF 1.x graphs are static and do not expose a dynamic "live tensor count".
    """
    # Input data
    filenames = [f"dummy_file_{i}.txt" for i in range(10)]

    with tf.compat.v1.Graph().as_default():
        # API Under Test: string_input_producer
        # This creates a FIFOQueue and enqueue ops to feed the pipeline
        queue = tf.compat.v1.train.string_input_producer(
            filenames,
            num_epochs=None,  # Run indefinitely to simulate training steps
            shuffle=False,
            capacity=32
        )

        # Consumer: Dequeue the strings
        filename = queue.dequeue()

        with tf.compat.v1.Session() as sess:
            # Initialize local variables (required for epochs counter)
            sess.run(tf.compat.v1.local_variables_initializer())

            # Start the queue runners
            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(coord=coord)

            # Start memory tracing
            tracemalloc.start()
            gc.collect()

            print("Starting loop...")
            # Reproduce the loop from the original bug report
            for step in range(300):
                # Execute the operation (analogous to forward pass)
                _ = sess.run(filename)

                # Log status periodically, mimicking the original output format
                if step % 50 == 0:
                    gc.collect()
                    current, peak = tracemalloc.get_traced_memory()
                    # Note: TF doesn't expose a "live tensor count" in the same way PyTorch does
                    # because the graph is static. We monitor memory allocation instead.
                    print(f"Step {step:3d} | Alloc: {current / 1e9:.3f}GB | Peak: {peak / 1e9:.3f}GB")

            # Stop resources
            coord.request_stop()
            coord.join(threads)
            tracemalloc.stop()

if __name__ == "__main__":
    test_string_input_producer_leak()