import time
import tensorflow as tf

# Define a function with a custom gradient, similar to the example in the API info
@tf.keras.ops.custom_gradient
def log1pexp(x):
    def grad(upstream):
        # Gradient expression for numerical stability
        return upstream * (1 - 1 / (1 + tf.exp(x)))
    return tf.math.log(1 + tf.exp(x)), grad

if __name__ == "__main__":
    # Preserve the original bug reproduction logic: large tensor dimensions and float16
    BS, VOCAB = 128, 8000000
    x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
    
    # Warmup
    for _ in range(10):
        _ = log1pexp(x)

    # Benchmark loop
    walltime = []
    for _ in range(100):
        s = time.time()
        _ = log1pexp(x)
        # In TensorFlow eager mode, execution is synchronous for the purpose of this high-level timing,
        # mimicking the behavior of the original script's synchronization.
        e = time.time()
        walltime.append(e - s)
    
    avg_time = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, custom_gradient latency: {1000 * avg_time}ms")