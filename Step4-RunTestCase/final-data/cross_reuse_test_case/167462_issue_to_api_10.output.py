import torch
import time
import tensorflow as tf

if __name__ == "__main__":
    # Parameters from the original bug report
    BS, VOCAB, K = 128, 8000000, 1000

    # Leverage the similar API: tf.distribute.get_replica_context
    # According to the API documentation, execution starts in the default replica context.
    # We verify this to ensure the environment is set up correctly before benchmarking.
    replica_context = tf.distribute.get_replica_context()
    assert replica_context is not None, "Expected a default replica context"
    print(f"Replica Context: {replica_context}")

    # Original bug reproduction logic adapted for TensorFlow
    # Using tf.math.top_k as the semantic equivalent of torch.topk
    x = tf.random.normal((BS, VOCAB), dtype=tf.float16)

    walltime = []
    for _ in range(100):
        s = time.time()
        # Perform the topk operation
        _ = tf.math.top_k(x, k=K)
        # TensorFlow eager execution is synchronous by default, 
        # acting similarly to torch.cuda.synchronize() for timing purposes in this context.
        e = time.time()
        walltime.append(e - s)

    # Discard warmup runs and calculate average
    walltime = walltime[10:]
    avg_latency = sum(walltime) / len(walltime)
    print(f"TensorFlow topk latency: {1000 * avg_latency}ms")