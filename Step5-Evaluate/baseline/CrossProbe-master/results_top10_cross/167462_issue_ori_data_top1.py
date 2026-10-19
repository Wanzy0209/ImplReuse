```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    # Conversion: torch.randn -> tf.random.normal
    # Note: TensorFlow uses GPU automatically if available and configured.
    x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
    walltime = []
    for _ in range(100):
        s = time.time()
        # Conversion: x.topk -> tf.math.top_k
        _ = tf.math.top_k(x, k=K)
        # Conversion: torch.cuda.synchronize
        # Note: TensorFlow 2.x eager execution does not have a direct public equivalent
        # to torch.cuda.synchronize(). To ensure accurate GPU timing, we force synchronization
        # by querying device memory info.
        _ = tf.config.experimental.get_memory_info('GPU:0')
        e = time.time()
        walltime.append(e-s)
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```