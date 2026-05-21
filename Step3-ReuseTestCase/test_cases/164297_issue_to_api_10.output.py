import torch
import tensorflow as tf

def test_value_context_numeric_conversion():
    """
    Test case for tf.distribute.experimental.ValueContext inspired by 
    PyTorch Issue #164297.
    
    The original issue involved a segfault when __int__ was called on a 
    C++ bound object (torch.onnx.OperatorExportTypes). This test verifies 
    that the similar API (ValueContext) handles integer conversion of its 
    properties correctly without crashing, ensuring the underlying C++ 
    integration (if any) or property accessors are stable.
    """
    # Instantiate ValueContext with specific integer properties
    # This mirrors the "Directly constructed" usage pattern.
    context = tf.distribute.experimental.ValueContext(
        replica_id_in_sync_group=2,
        num_replicas_in_sync=4
    )

    # The original bug occurred during a call to __int__ (dispatcher).
    # We explicitly test integer conversion of the context properties.
    # This exercises the numeric protocol similar to the failing PyTorch code.
    replica_id_int = int(context.replica_id_in_sync_group)
    num_replicas_int = int(context.num_replicas_in_sync)

    # Verify the conversion results and arithmetic operations
    # (Arithmetic was part of the usage example for the similar API)
    assert replica_id_int == 2, "replica_id_in_sync_group should be 2"
    assert num_replicas_int == 4, "num_replicas_in_sync should be 4"
    
    # Perform the calculation shown in the API documentation
    result = replica_id_int / num_replicas_int
    assert result == 0.5, "Calculation result should be 0.5"

if __name__ == "__main__":
    test_value_context_numeric_conversion()