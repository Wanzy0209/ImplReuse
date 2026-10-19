```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: device="cuda" -> TF uses GPU by default if available
    # Conversion: dtype=torch.float16 -> dtype=tf.float16
    x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
    walltime = []
    for _ in range(100):
        s = time.time()
        # Conversion: x.topk -> tf.math.top_k
        _ = tf.math.top_k(x, k=K)
        # Conversion: torch.cuda.synchronize -> tf.raw_ops.StreamSynchronize
        # Note: TF eager execution is asynchronous. We use StreamSynchronize to match PyTorch behavior.
        tf.raw_ops.StreamSynchronize()
        e = time.time()
        walltime.append(e-s)
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    # Conversion: torch.__version__ -> tf.__version__
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```