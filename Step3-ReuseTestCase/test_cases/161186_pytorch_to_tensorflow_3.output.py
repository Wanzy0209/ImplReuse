import torch
import tensorflow as tf
import sys

def test_tf_while_loop_memory():
    """
    Adapted test case for tf.while_loop based on PyTorch checkpoint memory leak.
    
    Original Bug: Memory leak in PyTorch when a custom autograd Function is used 
    with torch.utils.checkpoint.checkpoint due to early-stopping exceptions 
    preventing cleanup.
    
    Target API: tf.while_loop
    Goal: Verify that tf.while_loop handles custom gradients and memory management 
    correctly without leaking memory during repeated iterations.
    """
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPU available.")
        return

    # Configure memory growth to observe allocation dynamically
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

    # Define a custom gradient function similar to MyOp
    @tf.custom_gradient
    def my_op(inp):
        # Create large tensors to stress memory (2**20 floats ~ 4MB)
        out_0 = tf.zeros(2**20, dtype=tf.float32)
        out_1 = tf.zeros(2**20, dtype=tf.float32)

        def grad(d_out_0, d_out_1):
            # Mimic accessing saved tensors
            _ = out_0
            _ = out_1
            return None # No gradient for input

        return out_0, grad

    # Define the loop body for tf.while_loop
    def body(i):
        # Create a dummy input variable
        dummy_input = tf.Variable(tf.ones(2**20, dtype=tf.float32))
        
        with tf.GradientTape() as tape:
            # Apply the custom operation
            # Mimics: full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, ...)
            full_out = my_op(dummy_input)[0]
            loss = tf.reduce_sum(full_out)

        # Compute gradients (mimics .backward())
        _ = tape.gradient(loss, dummy_input)
        
        # In PyTorch, dummy_input.grad = None is called. 
        # In TF, GradientTape handles cleanup automatically unless persistent=True.
        
        return i + 1

    # Define the loop condition
    def cond(i):
        return i < 1000

    # Execute the loop
    # We wrap in tf.function to ensure graph execution, which is typical for 
    # performance and where complex memory management issues often surface.
    @tf.function
    def run_loop():
        return tf.while_loop(cond, body, [tf.constant(0)])

    print("Starting tf.while_loop test...")
    run_loop()
    print("Loop finished.")

    # Verify memory usage
    mem_info = tf.config.experimental.get_memory_info(gpus[0].name)
    current_mem_mb = mem_info['current'] / (1024**2)
    peak_mem_mb = mem_info['peak'] / (1024**2)
    
    print(f"Peak Memory Usage: {peak_mem_mb:.2f} MiB")
    print(f"Current Memory Usage: {current_mem_mb:.2f} MiB")

    # In the original PyTorch bug, memory would grow linearly with iterations 
    # (e.g., 1000 * 8MB = ~8GB leak).
    # We assert that memory usage remains reasonable (e.g., < 500MB) to ensure 
    # tf.while_loop does not suffer from a similar leak.
    # Note: 2**20 floats is 4MB. We create two tensors (out_0, out_1) = 8MB per iter.
    # 1000 iters * 8MB = 8GB potential leak.
    assert current_mem_mb < 500, f"Potential memory leak detected: {current_mem_mb:.2f} MiB"
    print("Test passed: No significant memory leak detected.")

if __name__ == "__main__":
    test_tf_while_loop_memory()