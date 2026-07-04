import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to simulate the "compiled" graph mode analogous to torch.compile
tf.compat.v1.disable_eager_execution()

def main():
    # 1. Setup Data
    # In the PyTorch case, data is generated on the fly. Here we use string_input_producer
    # to feed data into the graph, mimicking the input pipeline aspect.
    # We pass a list of strings representing numerical data to process.
    data_strings = tf.constant([f"{i}.0" for i in range(10)])
    
    # 2. Use the Target API: string_input_producer
    # This creates a queue to pipeline the data, similar to how DDP handles data distribution.
    # We set shuffle=False to match the deterministic nature of the PyTorch repro.
    queue = tf.compat.v1.train.string_input_producer(
        data_strings,
        num_epochs=3,  # Run for 3 epochs, similar to the loop in PyTorch code
        shuffle=False,
        capacity=32,
        name="input_queue"
    )

    # 3. Define Custom Function (Analogous to SimplistDoubleFn)
    # We use tf.py_func to define a custom operation, which mimics the custom autograd.Function
    # that caused issues in the PyTorch Dynamo backend.
    def custom_double_fn(x):
        # Simple operation: multiply by 2
        return x * 2.0

    # 4. Build the Model Graph
    # Dequeue the string and convert to float
    val_str = queue.dequeue()
    val = tf.strings.to_number(val_str, out_type=tf.float32)

    # Apply the custom function
    # Note: py_func is used here to represent a custom op that might not be standard,
    # testing the graph's ability to handle custom logic like torch.compile does.
    doubled = tf.py_func(custom_double_fn, [val], tf.float32)
    doubled.set_shape([]) # Restore shape info lost by py_func

    # Define Loss (MSE)
    # Target is the input value * 2, so loss should ideally be 0
    target = val * 2.0
    loss = tf.reduce_mean(tf.square(doubled - target))

    # Define Optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss)

    # 5. Execution Session
    with tf.compat.v1.Session() as sess:
        # Initialize local variables for epochs (required by string_input_producer)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Start QueueRunners to feed the queue (mimics the background threads in DDP)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

        print("Starting training loop...")
        try:
            step = 0
            # Run for a few iterations to verify stability
            while not coord.should_stop() and step < 10:
                _, loss_val = sess.run([train_op, loss])
                print(f"Step {step}, Loss: {loss_val:.6f}")
                step += 1
        except tf.errors.OutOfRangeError:
            print("Finished epochs (OutOfRangeError).")
        finally:
            # Stop the queue runners
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    main()