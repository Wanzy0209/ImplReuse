```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: device="cuda" -> tf.device('/GPU:0')
    # Conversion: dtype=torch.float16 -> dtype=tf.float16
    with tf.device('/GPU:0'):
        x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
        
    walltime = []
    for _ in range(100):
        s = time.time()
        
        # Conversion: x.topk(k=K, dim=-1) -> tf.math.top_k(x, k=K)
        # Note: tf.math.top_k operates on the last dimension by default
        _ = tf.math.top_k(x, k=K)
        
        # Conversion: torch.cuda.synchronize()
        # Note: TensorFlow eager execution is synchronous, so explicit synchronization 
        # is generally not required for timing in this manner.
        
        e = time.time()
        walltime.append(e-s)
        
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```