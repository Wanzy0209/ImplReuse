import torch
import tensorflow as tf

def test_enable_eager_execution_device_handling():
    """
    Adapted from PyTorch Issue 160909.
    
    Original Issue: torch.compile with PrivateUse1 sees tensors on "meta".
    The bug occurred because the compilation process used 'meta' tensors for tracing,
    which were then passed to custom device operations that expected real storage.
    
    This test verifies the behavior of the similar TensorFlow API 
    `tf.compat.v1.enable_eager_execution`. 
    Instead of compiling to a graph (which might involve abstract/meta tensors),
    enabling eager execution ensures operations are executed immediately with 
    concrete tensors on the specified device.
    """
    
    # 1. Enable eager execution.
    # This is the TensorFlow equivalent of setting the execution mode.
    # Unlike torch.compile (which creates a graph), this disables graph mode.
    # Note: This must be called at the start of the program.
    tf.compat.v1.enable_eager_execution()

    # 2. Define a simple model/function.
    # Equivalent to the Model() class in the PyTorch snippet.
    def my_model(x):
        return tf.multiply(x, 2)

    # 3. Prepare data on a specific device.
    # Equivalent to data.to('PrivateUse1').
    # We use '/cpu:0' as the target device.
    with tf.device('/cpu:0'):
        data = tf.constant([1.0, 2.0, 3.0])

    # 4. Execute the model.
    # In the PyTorch bug, this failed because 'meta' tensors were passed to the backend.
    # Here, we verify that eager execution passes concrete tensors to the operations.
    result = my_model(data)

    # 5. Assertions.
    # Verify that the result is a concrete EagerTensor (not a symbolic Tensor/Graph).
    assert isinstance(result, tf.EagerTensor), \
        "Expected EagerTensor, but got symbolic Tensor (Graph mode might still be active)."
    
    # Verify the computation is correct.
    expected = tf.constant([2.0, 4.0, 6.0])
    assert tf.reduce_all(tf.equal(result, expected)), "Computation result is incorrect."

    # Verify the device placement matches the input.
    assert result.device.endswith('/cpu:0'), f"Expected device /cpu:0, got {result.device}"

    print("Test passed: Eager execution handles device tensors correctly.")

if __name__ == "__main__":
    test_enable_eager_execution_device_handling()