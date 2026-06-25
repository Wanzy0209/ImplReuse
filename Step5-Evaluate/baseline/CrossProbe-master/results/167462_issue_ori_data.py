```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Conversion: torch.randn with device and dtype -> tf.random.normal inside device context
    with tf.device('/GPU:0'):
        x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
        
    walltime = []
    for _ in range(100):
        s = time.time()
        
        # Conversion: x.topk -> tf.math.top_k
        _ = tf.math.top_k(x, k=K)
        
        # Conversion: torch.cuda.synchronize() -> TF memory info query (forces sync)
        # Note: This is a common workaround to synchronize GPU execution in TF eager mode
        # without transferring data back to host.
        tf.config.experimental.get_memory_info('GPU:0')
        
        e = time.time()
        walltime.append(e-s)
        
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```