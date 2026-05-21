import tensorflow as tf
import collections

def test_session_run_values_in_traced_context():
    """
    Test case to verify that tf.compat.v1.train.SessionRunValues, which is 
    implemented as a collections.namedtuple, can be successfully instantiated 
    and used within a TensorFlow tracing context (tf.function).
    
    This test is derived from a PyTorch Dynamo bug where collections.defaultdict
    creation failed during tracing. It checks if the similar pattern (using a 
    collections type) works correctly in TensorFlow.
    """
    
    # Define a function that creates the SessionRunValues object.
    # This mirrors the logic in the PyTorch bug where a collections type
    # (defaultdict) was created inside a compiled function.
    @tf.function
    def create_run_values(results_tensor):
        # SessionRunValues is defined as:
        # collections.namedtuple("SessionRunValues", ["results", "options", "run_metadata"])
        return tf.compat.v1.train.SessionRunValues(
            results=results_tensor,
            options=None,
            run_metadata=None
        )

    # Create a simple tensor to pass as results
    tensor_input = tf.constant([1.0, 2.0, 3.0])

    # Execute the traced function
    # In the PyTorch bug, this step raised an Unsupported error for defaultdict.
    # Here we verify SessionRunValues (namedtuple) works.
    run_values = create_run_values(tensor_input)

    # Assertions to verify the object was created correctly and fields are accessible
    assert isinstance(run_values, tf.compat.v1.train.SessionRunValues)
    assert run_values.results is tensor_input
    assert run_values.options is None
    assert run_values.run_metadata is None
    
    print("Test passed: SessionRunValues created successfully in traced context.")

if __name__ == "__main__":
    test_session_run_values_in_traced_context()