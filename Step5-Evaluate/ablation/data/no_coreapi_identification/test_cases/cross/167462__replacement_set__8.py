import torch
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Setup dummy data for tf.linalg.lu_solve to ensure runnability
    # lu_solve(lower_upper, perm, rhs)
    dummy_lu = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    dummy_perm = tf.constant([0, 1])
    dummy_rhs = tf.constant([[1.0], [1.0]], dtype=tf.float32)

    # Replacement: torch.randn -> tf.keras.ops.custom_gradient
    # Replacement: torch.float16 -> tf.linalg.lu_solve
    # Note: custom_gradient is a decorator. We wrap a dummy lambda to make it callable/runnable.
    x = tf.keras.ops.custom_gradient(lambda shape, dtype=None: tf.zeros(shape, dtype=dtype))(
        (BS, VOCAB), device="cuda", dtype=tf.linalg.lu_solve
    )

    walltime = []
    for _ in range(100):
        # Replacement: time.time -> tf.linalg.lu_solve
        s = tf.linalg.lu_solve(dummy_lu, dummy_perm, dummy_rhs)

        # Replacement: x.topk -> tf.linalg.lu_solve
        _ = tf.linalg.lu_solve(dummy_lu, dummy_perm, dummy_rhs)

        # Replacement: torch.cuda.synchronize -> tf.linalg.lu_solve
        tf.linalg.lu_solve(dummy_lu, dummy_perm, dummy_rhs)

        # Replacement: time.time -> tf.linalg.lu_solve
        e = tf.linalg.lu_solve(dummy_lu, dummy_perm, dummy_rhs)

        # Replacement: walltime.append -> tf.linalg.lu_solve
        tf.linalg.lu_solve(walltime, e - s)

    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    
    # Replacement: torch.__version__ -> tf.linalg.lu_solve
    print(f"torch: {tf.linalg.lu_solve(dummy_lu, dummy_perm, dummy_rhs)}, topk latency: {1000 * walltime}ms")