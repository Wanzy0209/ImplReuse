```python
import time
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Conversion: torch.device("cuda") -> tf.device('/GPU:0')
    with tf.device('/GPU:0'):
        # Conversion: torch.randn -> tf.random.normal
        # Conversion: torch.float16 -> tf.float16
        x = tf.random.normal((BS, VOCAB), dtype=tf.float16)
        
    walltime = []
    for _ in range(100):
        s = time.time()
        
        # Conversion: x.topk(k=K, dim=-1) -> tf.math.top_k(x, k=K)
        # TF top_k operates on the last dimension by default
        values, _ = tf.math.top_k(x, k=K)
        
        # Conversion: torch.cuda.synchronize()
        # In TensorFlow eager mode, converting the tensor to numpy forces synchronization
        _ = values.numpy()
        
        e = time.time()
        walltime.append(e-s)
        
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"tensorflow: {tf.__version__}, topk latency: {1000 * walltime}ms")
```