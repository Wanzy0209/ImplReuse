import tensorflow as tf
import numpy as np

# Adapted from PyTorch bug report to test tf.VariableAggregation
# Original bug: Type mismatch (int64/bfloat16) causing eager/compile divergence.
# This test checks if tf.VariableAggregation handles these types correctly in tf.function.

def foo(arg0, arg1, arg2, sentinel):
    # arg0: int64 (4, 4)
    # arg1: int64 (5,)
    # arg2: bfloat16 (5000, 4)
    
    # Use Similar API: tf.VariableAggregation
    # We define the embedding weights with a specific aggregation strategy.
    # This tests the interaction of VariableAggregation with bfloat16 types.
    embedding_weights = tf.Variable(
        arg2, 
        dtype=tf.bfloat16, 
        aggregation=tf.VariableAggregation.MEAN
    )
    
    # Mimic original logic: tanh -> clamp -> embedding
    # Note: tf.tanh requires float, so we cast arg0 (int64) to bfloat16
    # to maintain the type stress test context.
    t0 = tf.cast(arg0, tf.bfloat16)
    t1 = tf.math.tanh(t0)
    
    # Clamp logic (indices must be int32 or int64)
    # We cast back to int64 for the embedding lookup
    indices = tf.cast(tf.clip_by_value(t1, 0, tf.cast(tf.shape(arg2)[0] - 1, tf.bfloat16)), tf.int64)
    
    # Embedding lookup
    t8 = tf.nn.embedding_lookup(embedding_weights, indices)
    
    # Min operation
    t9 = tf.reduce_min(t8)
    
    # Output with sentinel
    output = t9 + sentinel
    return output

# Inputs
arg0 = tf.constant(np.random.randint(0, 1000, (4, 4)), dtype=tf.int64)
arg1 = tf.constant(np.random.randint(0, 1000, (5,)), dtype=tf.int64)
arg2 = tf.constant(np.random.rand(5000, 4), dtype=tf.bfloat16)
sentinel = tf.constant(0.0, dtype=tf.bfloat16)

if __name__ == '__main__':
    print("Testing tf.VariableAggregation with bfloat16/int64 types...")
    
    # Eager Execution
    # We need to reset the variable for the graph run to be comparable if we were testing state,
    # but here we are testing the computation graph generation and type handling.
    # To make it clean, we instantiate the variable inside the function or use a fresh context.
    
    # Redefining foo to be stateless for the test or managing state carefully.
    # The original PyTorch test used `arg2` as weights.
    
    # Let's wrap the execution to handle the Variable creation cleanly.
    def run_test():
        # Recreate variable inside to ensure clean state for both runs
        # In a real distributed scenario, this would be a shared variable.
        weights = tf.Variable(arg2, dtype=tf.bfloat16, aggregation=tf.VariableAggregation.MEAN)
        
        def logic(w):
            t0 = tf.cast(arg0, tf.bfloat16)
            t1 = tf.math.tanh(t0)
            indices = tf.cast(tf.clip_by_value(t1, 0, tf.cast(tf.shape(arg2)[0] - 1, tf.bfloat16)), tf.int64)
            t8 = tf.nn.embedding_lookup(w, indices)
            t9 = tf.reduce_min(t8)
            return t9 + sentinel

        # Eager
        out_eager = logic(weights)
        
        # Graph (Compile)
        compiled_logic = tf.function(logic)
        out_compiled = compiled_logic(weights)
        
        return out_eager, out_compiled

    try:
        out_eager, out_compiled = run_test()
        
        # Check consistency
        if tf.reduce_all(tf.equal(out_eager, out_compiled)).numpy():
            print("Eager Success! ")
            print("Compile Success! ")
            print("Test Passed: No divergence.")
        else:
            print("Divergence detected! ")
            print(f"Eager: {out_eager}, Compiled: {out_compiled}")
            
    except Exception as e:
        print(f"Test Failed with error: {e}")
        import traceback
        traceback.print_exc()