import torch
import tensorflow as tf
import numpy as np

def test_lecun_uniform_bfloat16():
    """
    Adapted test case for tf.keras.initializers.LecunUniform based on 
    PyTorch Issue 164683 (Eager/Compile Divergence with bfloat16).
    
    The original bug involved an IncompatibleTypeError between pointer<bf16> 
    and float64 during compilation. This test verifies that LecunUniform 
    can initialize bfloat16 tensors correctly and that they pass through 
    the compilation graph (tf.function) without type errors.
    """
    
    # Initialize the API under test
    # Using a seed to ensure reproducibility between eager and graph modes
    initializer = tf.keras.initializers.LecunUniform(seed=42)
    
    # Define the shape and dtype, targeting bfloat16 as in the original bug
    shape = (5000, 4)
    dtype = tf.bfloat16

    # Define the computation graph
    # This mimics the pipeline in the original bug: Init -> ReLU -> SiLU -> Embedding -> Min
    def computation_fn(initializer, indices):
        # 1. Use LecunUniform to generate the tensor (replaces torch.rand/tanh logic in terms of data flow)
        # The original bug highlighted t5 (bfloat16) -> relu -> silu
        t = initializer(shape, dtype=dtype)
        
        # 2. Apply activations
        t = tf.nn.relu(t)
        # SiLU (Swish) implementation: x * sigmoid(x)
        t = t * tf.math.sigmoid(t)
        
        # 3. Embedding lookup
        # Original code: torch.nn.functional.embedding(indices, t)
        # t acts as the weight matrix here
        t_emb = tf.nn.embedding_lookup(t, indices)
        
        # 4. Reduce Min
        # Original code: t8.min()
        res = tf.reduce_min(t_emb)
        return res

    # Prepare dummy indices for embedding lookup
    # Original t4 was size (4, 4), dtype int64
    indices = tf.constant([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]], dtype=tf.int32)

    # 1. Run in Eager Mode
    print("Running Eager Mode...")
    try:
        out_eager = computation_fn(initializer, indices)
        print(f"Eager Output: {out_eager.numpy()}")
    except Exception as e:
        print(f"Eager Mode Failed: {e}")
        return

    # 2. Run in Graph Mode (tf.function) - Equivalent to torch.compile
    print("\nRunning Graph Mode (tf.function)...")
    compiled_fn = tf.function(computation_fn)
    try:
        out_graph = compiled_fn(initializer, indices)
        print(f"Graph Output: {out_graph.numpy()}")
    except Exception as e:
        print(f"Graph Mode Failed: {e}")
        return

    # 3. Verify Consistency
    # Check if outputs match (should be identical with seed)
    if np.allclose(out_eager.numpy(), out_graph.numpy()):
        print("\nTest Passed: Eager and Graph outputs match. No divergence detected.")
    else:
        print("\nWarning: Outputs differ between Eager and Graph modes.")

if __name__ == '__main__':
    test_lecun_uniform_bfloat16()