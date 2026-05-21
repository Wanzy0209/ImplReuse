import torch
import tensorflow as tf

def test_tf_name_scope_quantization_context():
    """
    Adapts the torch.compile regression test for mxfp8 quantization 
    to verify tf.name_scope behavior in a similar context.
    
    The original bug involves a performance regression when compiling 
    a quantization operation. Here, we verify that tf.name_scope correctly 
    encapsulates the operation, which is the structural equivalent of 
    wrapping a function for compilation/graph management.
    """
    # Dimensions from the original bug report (M 16384 K 16384)
    M, K = 16384, 16384
    
    # The original bug uses a specific mode: dim0_mxfp8_floor
    # We use this as the name for the scope to preserve the context.
    scope_name = "dim0_mxfp8_floor"

    # In the original bug, torch.compile wraps the operation.
    # Here, tf.name_scope wraps the operation to group it in the graph.
    with tf.name_scope(scope_name):
        # Create input tensor simulating the benchmark workload
        input_tensor = tf.random.normal((M, K), dtype=tf.float32)
        
        # Simulate the quantization/casting operation.
        # While TF core may not have the specific mxfp8 kernel, 
        # tf.cast represents the core logic of changing precision.
        quantized_tensor = tf.cast(input_tensor, tf.bfloat16)

    # Verify that the operation was captured within the scope.
    # This checks the primary behavior of tf.name_scope: naming hierarchy.
    assert scope_name in quantized_tensor.name, \
        f"Expected scope '{scope_name}' in tensor name '{quantized_tensor.name}'"

    # To better mirror torch.compile (which generates a graph), 
    # we also verify the behavior inside tf.function (graph mode).
    @tf.function
    def compiled_quantization_op():
        with tf.name_scope(scope_name):
            x = tf.random.normal((M, K), dtype=tf.float32)
            return tf.cast(x, tf.bfloat16)

    graph_result = compiled_quantization_op()
    assert scope_name in graph_result.name, \
        f"Expected scope '{scope_name}' in graph tensor name '{graph_result.name}'"

if __name__ == "__main__":
    test_tf_name_scope_quantization_context()
    print("Test passed: tf.name_scope correctly encapsulated the quantization logic.")