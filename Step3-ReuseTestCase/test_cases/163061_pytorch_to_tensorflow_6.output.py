import torch
import tensorflow as tf
import threading
import time

# Standard TensorFlow operation (analogous to torch_add)
def tf_add(x: tf.Tensor, y: tf.Tensor):
    return x + y

# Operation wrapped in the similar API: tf.compat.v1.name_scope
# (analogous to torch_compile_add in terms of being a wrapper/context)
def tf_name_scope_add(x: tf.Tensor, y: tf.Tensor):
    # Note: In TF 2.x eager mode, name_scope with skip_on_eager=True (default)
    # acts as a NullContext. We use it here to test the API behavior.
    with tf.compat.v1.name_scope("add_scope"):
        return x + y

# Helper to check GIL behavior via threading
# This is a heuristic to see if the main thread releases the GIL
# allowing other threads to run during the operation.
def check_gil_concurrency():
    worker_ran = False
    
    def worker():
        nonlocal worker_ran
        # Simulate some Python work
        _ = sum(range(1000000))
        worker_ran = True
        
    thread = threading.Thread(target=worker)
    thread.start()
    return thread

def main():
    # Check for GPU availability to match original context
    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    print(f"Using device: {device}")

    with tf.device(device):
        x = tf.random.normal((4096, 4096))
        y = tf.random.normal((4096, 4096))
        
        # Warmup
        _ = tf_add(x, y)
        _ = tf_name_scope_add(x, y)

        print("Starting loop...")
        for _ in range(10):
            # Standard op
            t = check_gil_concurrency()
            res1 = tf_add(x, y)
            t.join()
            
            # Op inside name_scope
            t = check_gil_concurrency()
            res2 = tf_name_scope_add(x, y)
            t.join()
            
            # Basic sanity check to ensure operations ran
            assert res1.shape == (4096, 4096)
            assert res2.shape == (4096, 4096)

if __name__ == "__main__":
    main()