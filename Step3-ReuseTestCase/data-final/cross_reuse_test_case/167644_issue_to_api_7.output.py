import torch
import tensorflow as tf
import warnings

def test_hierarchical_copy_all_reduce_warnings():
    """
    Test case to check if using tf.distribute.HierarchicalCopyAllReduce
    triggers warning spam about internal behavior or deprecations, similar
    to the issue reported for torch.set_float32_matmul_precision.
    """
    
    # Capture warnings to detect potential spam
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Initialize the strategy with the specific cross device op
        # This mirrors the usage of torch.set_float32_matmul_precision
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )

        # Run a simple step to trigger the internal logic of the cross device op
        @tf.function
        def step():
            return tf.constant(1.0)

        strategy.run(step)

        # Analyze captured warnings
        if w:
            print(f"Detected {len(w)} warning(s):")
            for warning in w:
                print(f" - {warning.category.__name__}: {warning.message}")
        else:
            print("No warnings detected.")

if __name__ == '__main__':
    test_hierarchical_copy_all_reduce_warnings()