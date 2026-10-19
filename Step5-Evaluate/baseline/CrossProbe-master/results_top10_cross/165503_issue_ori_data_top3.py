```python
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    B = 8
    C = 2
    P = 4
    R = 256
    L = 64
    target_num = 10

    # Conversion: torch.randn -> tf.random.normal
    y_chirp = tf.random.normal((target_num, C, P, L))
    # Conversion: torch.randint -> tf.random.uniform (dtype int64 to match PyTorch default)
    b0 = tf.random.uniform((target_num,), minval=0, maxval=B, dtype=tf.int64)
    r_int = tf.random.uniform((target_num,), minval=0, maxval=R, dtype=tf.int64)
    # device is implicit in TensorFlow

    # way 1
    # Conversion: torch.zeros -> tf.Variable(tf.zeros(...)) to allow in-place updates
    y_loop = tf.Variable(tf.zeros((B, C, P, R + L), dtype=y_chirp.dtype))
    for i in range(target_num):
        y_chirp_tar_i = y_chirp[i]
        r_start = r_int[i]
        # Conversion: torch.arange -> tf.range
        r_indices = tf.range(r_start, r_start + L)
        
        # Conversion: Slice assignment with addition
        # TF Variables support slice assignment via .assign()
        # We read the current slice, add the new values, and assign back.
        current_slice = y_loop[b0[i], :, :, r_indices]
        y_loop[b0[i], :, :, r_indices].assign(current_slice + y_chirp_tar_i)

    # way 2
    y_vec = tf.Variable(tf.zeros((B, C, P, R + L), dtype=y_chirp.dtype))
    N = target_num
    
    # Conversion: view/expand -> reshape/broadcast_to
    b_idx = tf.broadcast_to(tf.reshape(b0, [N, 1, 1, 1]), [N, C, P, L])
    c_idx = tf.broadcast_to(tf.reshape(tf.range(C, dtype=tf.int64), [1, C,