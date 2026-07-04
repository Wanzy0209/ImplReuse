import tensorflow as tf
import sys

# The core bug reproduction logic involves passing an extremely large integer
# (9223372036854775803, close to INT64_MAX) to a parameter that defines a size or count.
# In the original PyTorch bug, this was passed to the 'padding' parameter of LazyConv1d.
# Here, we adapt this logic to the 'num_workers' parameter of create_local_cluster.

def test_large_integer_input():
    large_int = 9223372036854775803
    
    try:
        # Attempt to create a cluster with an impossibly large number of workers.
        # This tests if the API handles the large integer gracefully (e.g., raises an error)
        # or crashes (similar to the segmentation fault in the PyTorch issue).
        workers, ps = tf.test.create_local_cluster(
            num_workers=large_int,
            num_ps=1
        )
        print("Test passed: API handled the input without crashing (unexpected success).")
    except Exception as e:
        # We expect a standard Python/TensorFlow exception (e.g., ResourceExhausted, InvalidArgument)
        # rather than a segmentation fault.
        print(f"Test passed: Caught expected exception {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_large_integer_input()