import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

def test_get_source_inputs_with_tensor_views():
    """
    Adapted from PyTorch Issue #161324.
    
    Original Bug: Data inconsistencies when using batch_isend_irecv with 2D tensor views.
    The issue occurred when transferring data from slices of a tensor.
    
    Adaptation: Since tf.keras.utils.get_source_inputs is a graph introspection API
    rather than a distributed communication API, we verify that the API correctly
    identifies the source inputs when dealing with tensor "views" (slices).
    This ensures there are no inconsistencies in the graph lineage when using
    sliced tensors, analogous to the data corruption in the original bug.
    """
    
    # Setup parameters mimicking the PyTorch bug report
    batch_size = 8
    total_columns = 16
    # Define offsets to split the tensor into two non-contiguous or distinct views
    split_offsets = [0, 4, 8, 12]

    # Mimic "local_tensor" on the Sender rank (Rank 0)
    # In Keras, this is the Input layer/tensor
    local_tensor = Input(shape=(total_columns,), batch_size=batch_size, name="local_tensor")

    # Mimic the slicing logic: 
    # dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
    # dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]
    # We use Lambda layers to create tensor views/slices
    dst_t1 = Lambda(lambda x: x[:, split_offsets[0]:split_offsets[1]], name="dst_t1")(local_tensor)
    dst_t2 = Lambda(lambda x: x[:, split_offsets[2]:split_offsets[3]], name="dst_t2")(local_tensor)

    # The core logic: Verify that get_source_inputs correctly traces back
    # from the views (slices) to the original source tensor.
    # This checks for "inconsistencies" in the graph lineage, analogous to
    # data inconsistencies in the PyTorch distributed bug.

    # Check source for the first view
    source_inputs_1 = tf.keras.utils.get_source_inputs(dst_t1)
    assert source_inputs_1 is not None, "Source inputs for dst_t1 should not be None"
    assert len(source_inputs_1) == 1, f"Expected one source input for dst_t1, got {len(source_inputs_1)}"
    assert source_inputs_1[0] == local_tensor, "Source input mismatch for dst_t1: View did not trace back to original tensor"

    # Check source for the second view
    source_inputs_2 = tf.keras.utils.get_source_inputs(dst_t2)
    assert source_inputs_2 is not None, "Source inputs for dst_t2 should not be None"
    assert len(source_inputs_2) == 1, f"Expected one source input for dst_t2, got {len(source_inputs_2)}"
    assert source_inputs_2[0] == local_tensor, "Source input mismatch for dst_t2: View did not trace back to original tensor"

    print("Test Passed: get_source_inputs correctly handles tensor views/slices without inconsistencies.")

if __name__ == "__main__":
    test_get_source_inputs_with_tensor_views()