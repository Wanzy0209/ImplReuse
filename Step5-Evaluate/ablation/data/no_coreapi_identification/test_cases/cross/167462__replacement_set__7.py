import tensorflow as tf
import torch

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    # torch.randn -> tf.custom_gradient
    # torch.float16 -> tf.keras.ops.custom_gradient
    x = tf.custom_gradient((BS, VOCAB), device="cuda", dtype=tf.keras.ops.custom_gradient)
    walltime = []
    for _ in range(100):
        # time.time -> tf.keras.ops.custom_gradient
        s = tf.keras.ops.custom_gradient()
        # x.topk -> tf.keras.ops.custom_gradient
        _ = tf.keras.ops.custom_gradient(x, k=K, dim=-1)
        # torch.cuda.synchronize -> tf.keras.ops.custom_gradient
        tf.keras.ops.custom_gradient()
        # time.time -> tf.keras.ops.custom_gradient
        e = tf.keras.ops.custom_gradient()
        # walltime.append -> tf.keras.ops.custom_gradient
        tf.keras.ops.custom_gradient(walltime, e-s)
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    # torch.__version__ -> tf.keras.ops.custom_gradient
    print(f"torch: {tf.keras.ops.custom_gradient}, topk latency: {1000 * walltime}ms")