import torch
import tensorflow as tf
import threading
import time
import sys

# Helper class to verify if the GIL is released during execution
class GILChecker:
    def __init__(self):
        self.counter = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker)

    def _worker(self):
        """Background thread that increments a counter while GIL is held."""
        while not self.stop_event.is_set():
            self.counter += 1
            # A tiny sleep to yield, but if GIL is held by main thread, 
            # this thread won't run much.
            time.sleep(0.00001)

    def start(self):
        self.counter = 0
        self.stop_event.clear()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.counter

def check_gil_released(func, *args, **kwargs):
    """
    Executes the given function and checks if the GIL was released
    by monitoring a background thread.
    """
    checker = GILChecker()
    
    # Warmup run to ensure compilation/initialization overhead doesn't skew results
    func(*args, **kwargs)
    
    checker.start()
    start_time = time.time()
    
    # Execute the function under test
    result = func(*args, **kwargs)
    
    end_time = time.time()
    bg_work = checker.stop()
    
    duration = end_time - start_time
    print(f"Function: {func.__name__} | Duration: {duration:.4f}s | Background Thread Increments: {bg_work}")
    
    # If GIL is held, the background thread will barely increment (likely 0 or very low).
    # If GIL is released, the background thread will run concurrently and increment significantly.
    # We assert that some background work happened, implying GIL was released.
    assert bg_work > 100, f"GIL appears to be held! Background thread only incremented {bg_work} times during execution."
    
    return result

# --- Adapted Test Case Functions ---

def tf_standard_add(x: tf.Tensor, y: tf.Tensor):
    """Standard TensorFlow addition."""
    return x + y

def tf_name_scope_add(x: tf.Tensor, y: tf.Tensor):
    """
    Addition performed inside tf.keras.name_scope.
    This is the API identified as similar to torch.compile in the context of this issue.
    We verify that using this scope does not inadvertently hold the GIL.
    """
    with tf.keras.name_scope("scope_add"):
        return x + y

def main():
    # Check for GPU availability, fallback to CPU if necessary
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    print(f"Running on device: {device_name}")

    with tf.device(device_name):
        # Create large tensors to ensure the operation takes enough time to measure GIL behavior
        # Matching the scale of the original PyTorch test case (4096x4096)
        x = tf.random.normal((4096, 4096))
        y = tf.random.normal((4096, 4096))

        print("\n--- Testing GIL Release ---")
        
        # Test 1: Standard operation
        print("\n1. Testing standard TF operation (baseline):")
        check_gil_released(tf_standard_add, x, y)

        # Test 2: Operation inside tf.keras.name_scope (The Similar API)
        print("\n2. Testing operation inside tf.keras.name_scope:")
        check_gil_released(tf_name_scope_add, x, y)

    print("\nAll tests passed. GIL is being released appropriately.")

if __name__ == "__main__":
    main()