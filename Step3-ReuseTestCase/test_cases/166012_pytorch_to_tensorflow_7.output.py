import torch
import tensorflow as tf
from tf.keras.backend import name_scope

def test_name_scope_consistency_on_cache_hit():
    """
    Test to verify that tf.keras.backend.name_scope produces consistent
    operation names (entries) between the initial trace (cache miss) and
    subsequent calls (cache hit), similar to the expected behavior of
    torch.compile logging entries.
    """
    
    # Define a model using tf.function to simulate the compilation/caching behavior
    @tf.function
    def simple_model(x):
        # Use name_scope to create a context for operations
        with name_scope("my_test_scope"):
            # Create an operation that should be scoped
            y = x * 2
            z = y + 1
        return z

    # 1. First call: Cache Miss (Tracing)
    # This triggers the graph construction and applies the name_scope
    input_tensor = tf.constant(1.0)
    result_miss = simple_model(input_tensor)
    
    # Get the concrete function and graph generated during the miss
    concrete_func_miss = simple_model.get_concrete_function(input_tensor)
    graph_miss = concrete_func_miss.graph
    
    # Collect "entries" (operation names) from the first trace
    ops_names_miss = [op.name for op in graph_miss.get_operations()]
    
    # 2. Second call: Cache Hit (Reusing the graph)
    # This should reuse the previously traced graph
    input_tensor_2 = tf.constant(2.0)
    result_hit = simple_model(input_tensor_2)
    
    # Get the concrete function and graph used during the hit
    concrete_func_hit = simple_model.get_concrete_function(input_tensor_2)
    graph_hit = concrete_func_hit.graph
    
    # Collect "entries" (operation names) from the cache hit
    ops_names_hit = [op.name for op in graph_hit.get_operations()]

    # 3. Verification
    # Check that the scope name "my_test_scope" is present in the operations
    # for both the miss and hit scenarios.
    scope_prefix = "my_test_scope"
    
    # Assert presence in cache miss
    assert any(scope_prefix in name for name in ops_names_miss), \
        f"Name scope '{scope_prefix}' missing in cache miss graph entries: {ops_names_miss}"
        
    # Assert presence in cache hit
    # This corresponds to checking if the "tlparse entries" are consistent.
    # If the scope was missing in the hit but present in the miss, it would be a bug.
    assert any(scope_prefix in name for name in ops_names_hit), \
        f"Name scope '{scope_prefix}' missing in cache hit graph entries: {ops_names_hit}"

    # Assert that the list of operations is identical (consistency check)
    assert ops_names_miss == ops_names_hit, \
        f"Inconsistency detected: Graph entries differ between cache miss and hit.\nMiss: {ops_names_miss}\nHit: {ops_names_hit}"

    print("Test passed: Name scope entries are consistent between cache miss and hit.")

if __name__ == "__main__":
    test_name_scope_consistency_on_cache_hit()