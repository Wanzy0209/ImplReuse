import time
import tensorflow as tf

# Define a function with a custom gradient, leveraging the similar API.
# We wrap a heavy operation (tf.math.top_k) to mirror the context of the original bug report.
@tf.custom_gradient
def custom_topk_op(x):
    k = 1000
    # Forward pass: Perform topk
    values, indices = tf.math.top_k(x, k=k)
    
    # Define the gradient function
    def grad(upstream):
        # Simplified gradient logic for demonstration: 
        # In a real scenario, this would scatter the upstream gradients back to the 
        # original indices. Here we return zeros to keep the test minimal and runnable.
        return tf.zeros_like(x)
    
    return values, grad

if __name__ == "__main__":
    # Reproduce the setup from the original bug report
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Use float16 to match the original bug report's dtype
    x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
    
    # Warm-up runs (similar to the original script ignoring the first 10 measurements)
    for _ in range(10):
        _ = custom_topk_op(x)
        
    walltime = []
    # Benchmark loop
    for _ in range(100):
        s = time.time()
        _ = custom_topk_op(x)
        # Note: TensorFlow eager execution is synchronous by default, 
        # so explicit synchronization is usually not required like in PyTorch.
        e = time.time()
        walltime.append(e - s)
        
    # Calculate average latency, excluding warm-up
    walltime = walltime[10:]
    avg_walltime = sum(walltime) / len(walltime)
    
    print(f"tensorflow: {tf.__version__}, custom_topk_op latency: {1000 * avg_walltime}ms")