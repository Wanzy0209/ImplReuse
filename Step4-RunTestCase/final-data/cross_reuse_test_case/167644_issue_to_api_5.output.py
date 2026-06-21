import torch
import tensorflow as tf
import warnings

# Define a custom BatchableExtensionType based on the similar API documentation
class Vehicle(tf.experimental.BatchableExtensionType):
    """An ExtensionType that can be batched and unbatched."""
    top_speed: tf.Tensor
    mpg: tf.Tensor

def test_batchable_extension_warnings():
    """
    Test that using the public BatchableExtensionType API does not trigger
    internal deprecation warnings or warning spam.
    
    This mirrors the PyTorch issue where calling a public API 
    (torch.set_float32_matmul_precision) resulted in unavoidable warnings 
    about internal implementation details.
    """
    
    # Create a batch of data using the public API
    batch = Vehicle(tf.constant([120, 150, 80]), tf.constant([30, 40, 12]))

    # Capture warnings to check for "spam" similar to the reported issue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Perform an operation that relies on the internal batching mechanism
        # This is analogous to setting precision in PyTorch which triggers internal checks
        result = tf.map_fn(
            lambda vehicle: vehicle.top_speed * vehicle.mpg,
            batch,
            fn_output_signature=tf.int32
        )

        # Assert that no UserWarnings or DeprecationWarnings were raised
        # by the internal implementation during standard usage.
        filtered_warnings = [warning for warning in w 
                             if issubclass(warning.category, (UserWarning, DeprecationWarning))]
        
        if len(filtered_warnings) > 0:
            for warning in filtered_warnings:
                print(f"Caught Warning: {warning.category.__name__}: {warning.message}")
            
            raise AssertionError(
                f"Usage of BatchableExtensionType triggered {len(filtered_warnings)} "
                "internal warning(s). This mimics the behavior of the PyTorch bug "
                "where internal deprecations leaked to the user."
            )

    print("Test passed: No internal warnings detected during API usage.")

if __name__ == '__main__':
    test_batchable_extension_warnings()