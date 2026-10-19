```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Conversion: torch.randn with device and dtype -> tf.random.normal inside device context
    # Note: Using standard tf.random.normal instead of internal test_util.rand for valid public API usage.
    with tf.device('/GPU:0'):
        x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
    
    walltime = []
    for _ in range(100):
        s = time.time()
        # Conversion: x.topk -> tf.math.top_k
        # Note: TensorFlow eager execution is synchronous by default, acting as an implicit synchronize.
        _ = tf.math.top_k(x, k=K)
        # torch.cuda.synchronize() is not needed in TF eager mode as the op blocks execution.
        e = time.time()
        walltime.append(e-s)
    
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```