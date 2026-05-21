import torch
import tensorflow as tf
import os
import shutil

def test_depth_to_space_with_profiling():
    """
    Adapted test case for tf.compat.v1.nn.depth_to_space based on the 
    torch.profiler.profile bug reproduction logic.
    
    Original Logic:
    1. Setup Profiler with trace handler.
    2. Start Profiler.
    3. Execute operations (math ops).
    4. Stop Profiler.
    
    Adapted Logic:
    1. Setup TF Profiler log directory.
    2. Start TF Profiler.
    3. Execute the similar API (tf.compat.v1.nn.depth_to_space).
    4. Stop TF Profiler.
    5. Verify the operation output.
    """
    
    # Setup: Define log directory (mimicking on_trace_ready setup)
    log_dir = "./tf_trace_logs"
    worker_name = "trace"
    
    # Clean up previous runs if necessary
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # Start Profiler (mimicking profile.start())
    # Note: tf.profiler.experimental.start saves traces to log_dir
    tf.profiler.experimental.start(log_dir)

    # Prepare input data
    # depth_to_space requires input depth to be divisible by block_size^2
    block_size = 2
    batch_size = 1
    height = 2
    width = 2
    depth = block_size * block_size  # depth must be 4 for block_size 2
    
    # Create random tensor on CPU (mimicking torch.randn)
    # Using NHWC format as per default for depth_to_space
    x = tf.random.normal([batch_size, height, width, depth], name="input_tensor")

    # Execute the Similar API (mimicking the workload: y = x + 21, z = x * 15)
    # We are testing tf.compat.v1.nn.depth_to_space
    try:
        y = tf.compat.v1.nn.depth_to_space(
            input=x,
            block_size=block_size,
            data_format="NHWC",
            name="depth_to_space_op"
        )
        
        # Verify behavior: Check output shape
        # Expected shape: [batch, height*block, width*block, depth/(block*block)]
        expected_shape = [batch_size, height * block_size, width * block_size, 1]
        assert list(y.shape) == expected_shape, \
            f"Shape mismatch. Expected {expected_shape}, got {list(y.shape)}"
            
    except Exception as e:
        print(f"Error during operation execution: {e}")
        raise
    finally:
        # Stop Profiler (mimicking profile.stop())
        # This triggers the trace saving, analogous to the bug scenario
        tf.profiler.experimental.stop()

    print(f"Test completed successfully. Trace saved to {log_dir}")
    print(f"Output shape verified: {y.shape}")

if __name__ == "__main__":
    test_depth_to_space_with_profiling()