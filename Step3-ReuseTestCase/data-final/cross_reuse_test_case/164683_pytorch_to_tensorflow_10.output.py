import torch
import tensorflow as tf
import numpy as np

# Setup: Set a step for the API under test
tf.summary.experimental.set_step(42)

def foo(arg0, arg1, arg2, sentinel):
    # API Under Test: tf.summary.experimental.get_step
    # We retrieve the step value here to test its behavior within the graph
    step_val = tf.summary.experimental.get_step()
    
    # Replicating the logic flow from the original PyTorch test case
    # t0 = arg0 (int64)
    t0 = arg0
    
    # t1 = torch.tanh(t0)
    # PyTorch allows tanh on int64 (implicit cast). TF requires explicit cast for tanh.
    t1 = tf.tanh(tf.cast(t0, tf.float32))
    
    t2 = arg1
    t3 = tf.reduce_min(t2)
    
    # t4 = t1.clone(); t4.fill_(t3.item())
    # Create a tensor filled with the min value
    t4 = tf.fill(tf.shape(t1), tf.cast(t3, tf.float32))
    
    t5 = arg2
    t6 = tf.nn.relu(t5)
    t7 = tf.nn.silu(t6) # SiLU activation
    
    # t8 = torch.nn.functional_embedding(...)
    # Clamp indices and cast to int32 for embedding lookup
    indices = tf.clip_by_value(tf.cast(t4, tf.int32), 0, tf.shape(t7)[0] - 1)
    t8 = tf.nn.embedding_lookup(t7, indices)
    
    t9 = tf.reduce_min(t8)
    
    # Original: output = t9 + sentinel
    # Adaptation: Use the step_val from get_step to test type interaction
    # step_val is an int, t9 is bfloat16. This tests type promotion/casting logic.
    output = t9 + tf.cast(step_val, tf.bfloat16)
    return output

# Create inputs matching the original test case dimensions
arg0 = tf.constant(np.random.randint(0, 1000, (4, 4)), dtype=tf.int64)
arg1 = tf.constant(np.random.randint(0, 1000, (5,)), dtype=tf.int64)
arg2 = tf.Variable(tf.random.uniform((5000, 4), dtype=tf.bfloat16))
sentinel = tf.constant(0.0, dtype=tf.bfloat16)

if __name__ == '__main__':
    # Eager Execution
    out_eager = foo(arg0, arg1, arg2, sentinel)
    print(f'Eager Success!  Output: {out_eager.numpy()}')

    # Graph Execution (Compile equivalent)
    compiled_foo = tf.function(foo)
    out_graph = compiled_foo(arg0, arg1, arg2, sentinel)
    print(f'Graph Success!  Output: {out_graph.numpy()}')

    # Assertion to check for Eager/Compile divergence
    # Using np.allclose to handle potential minor floating point differences
    assert np.allclose(out_eager.numpy(), out_graph.numpy()), \
        f"Divergence detected: Eager={out_eager.numpy()}, Graph={out_graph.numpy()}"
    print('Test Passed: Eager and Graph outputs match.')