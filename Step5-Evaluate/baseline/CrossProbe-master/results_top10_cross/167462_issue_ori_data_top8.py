```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: device="cuda" -> TF automatically places tensors on GPU if available
    # Conversion: dtype=torch.float16 -> dtype=tf.float16
    x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
    walltime = []
    for _ in range(100):
        s = time.time()
        # Conversion: x.topk(k=K, dim=-1) -> tf.math.top_k(x, k=K)
        # Note: tf.math.top_k defaults to the last axis
        _ = tf.math.top_k(x, k=K)
        # Conversion: torch.cuda.synchronize()
        # In TensorFlow 2.x eager execution, operations are synchronous by default.
        # The Python interpreter waits for the operation to finish before proceeding.
        e = time.time()
        walltime.append(e-s)
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```