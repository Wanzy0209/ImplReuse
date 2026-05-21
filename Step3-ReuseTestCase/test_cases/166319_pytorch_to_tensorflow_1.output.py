import torch
import tensorflow as tf
import numpy as np

# Set seed for reproducibility
tf.random.set_seed(1215252001)

def computation_fn(arg_0, arg_1, arg_2):
    """
    Adapted from the PyTorch fuzzed_program.
    This function will be executed in parallel on TPU shards.
    """
    # arg_0: size=(9, 1, 15, 4) -> Assuming shard size matches this
    # PyTorch: var_node_3 = torch.squeeze(var_node_4)
    var_node_3 = tf.squeeze(arg_0)  # size=(9, 15, 4)

    # PyTorch: var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    # tf.split returns a list, we take the first element
    var_node_2 = tf.split(var_node_3, 4, axis=2)[0]  # size=(9, 15, 1)

    # PyTorch: var_node_1 = torch.squeeze(var_node_2)
    var_node_1 = tf.squeeze(var_node_2)  # size=(9, 15)

    # arg_1: size=(20, 15) -> Assuming shard size matches this
    # PyTorch: _input_size_var_node_6 = var_node_7.size(0)
    input_size = tf.shape(arg_1)[0]

    # PyTorch: _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), ...)
    # Note: In graph mode/TPU, random ops must be stateless or handled correctly.
    # We use tf.random.uniform.
    indices = tf.random.uniform(
        (18, 15), 
        minval=0, 
        maxval=tf.cast(input_size, tf.int32), 
        dtype=tf.int32
    )

    # PyTorch: var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    # TensorFlow equivalent for torch.gather(input, 0, index) where index is 2D
    # is tf.gather_nd with constructed coordinates.
    # We need to gather elements at (indices[i, j], j) for each i, j.
    # Create column indices (0..14) tiled for 18 rows.
    col_indices = tf.tile(tf.range(15)[tf.newaxis, :], [18, 1])
    # Stack to get (18, 15, 2) coordinates
    coords = tf.stack([indices, col_indices], axis=-1)
    var_node_6 = tf.gather_nd(arg_1, coords)  # size=(18, 15)

    # PyTorch: var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
    var_node_5 = tf.split(var_node_6, 2, axis=0)[0]  # size=(9, 15)

    # PyTorch: var_node_0 = torch.mul(var_node_1, var_node_5)
    var_node_0 = tf.multiply(var_node_1, var_node_5)  # size=(9, 15)

    return var_node_0

def run_test():
    # To use tf.compat.v1.tpu.batch_parallel, we need to initialize the TPU system.
    # Since this is a generated test case, we attempt to resolve a TPU.
    # If no TPU is available, we fall back to CPU execution using tf.function(jit_compile=True)
    # to mimic the compilation aspect of the original bug report.
    
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tpu_strategy = tf.distribute.TPUStrategy(resolver)
        use_tpu = True
        print(" TPU initialized")
    except ValueError:
        print(" TPU not found, falling back to CPU/XLA compilation")
        use_tpu = False
        # Create a dummy strategy for CPU
        tpu_strategy = tf.distribute.get_strategy()

    # Define inputs.
    # The original PyTorch code operates on specific shapes: (9, 1, 15, 4), (20, 15), (18, 15).
    # batch_parallel splits inputs along dim 0.
    # If we use num_shards=2, the global inputs should be double the shard size.
    # Shard sizes (target shapes inside computation_fn):
    # arg_0: (9, 1, 15, 4) -> Global: (18, 1, 15, 4)
    # arg_1: (20, 15)      -> Global: (40, 15)
    # arg_2: (18, 15)      -> Global: (36, 15)
    
    num_shards = 2
    
    # Generate random data matching the types (int32, int64)
    # PyTorch used torch.randint(5, 30, ...)
    global_arg_0 = tf.random.uniform((18, 1, 15, 4), minval=5, maxval=30, dtype=tf.int32)
    global_arg_1 = tf.random.uniform((40, 15), minval=5, maxval=30, dtype=tf.int64)
    global_arg_2 = tf.random.uniform((36, 15), minval=5, maxval=30, dtype=tf.int64)

    inputs = [global_arg_0, global_arg_1, global_arg_2]

    if use_tpu:
        # Execute on TPU using batch_parallel
        # batch_parallel expects the computation to return the outputs, which it concatenates.
        with tpu_strategy.scope():
            result = tf.compat.v1.tpu.batch_parallel(
                computation_fn,
                inputs=inputs,
                num_shards=num_shards
            )
        
        # Expected output shape: (9 * 2, 15) = (18, 15)
        expected_shape = (18, 15)
    else:
        # Fallback: Compile and run on CPU to verify logic
        # We manually split inputs to simulate sharding for the logic check
        shard_0_inputs = [t[:t.shape[0]//2] for t in inputs]
        shard_1_inputs = [t[t.shape[0]//2:] for t in inputs]
        
        # Use XLA compilation to mimic torch.compile
        compiled_fn = tf.function(computation_fn, jit_compile=True)
        
        res_0 = compiled_fn(*shard_0_inputs)
        res_1 = compiled_fn(*shard_1_inputs)
        
        # Concatenate results manually
        result = tf.concat([res_0, res_1], axis=0)
        expected_shape = (18, 15)

    # Verify output shape
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape} != {expected_shape}"
    print(f" Test passed. Output shape: {result.shape}")

if __name__ == "__main__":
    run_test()