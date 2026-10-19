```python
import numpy as np
import tensorflow as tf


def test_non_deterministic():
    gt_res = np.array(
        [
            [
                [-99.0, 1.0, -97.0, 3.0],
                [-96.0, -98.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            [
                [-90.0, -92.0, 14.0, 15.0],
                [-93.0, 17.0, -91.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        ],
        dtype=np.float32,
    )
    gt_input_grad = np.array(
        [
            [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    gt_src_grad = np.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    # Helper function to mimic torch.scatter
    # PyTorch scatter: self[index[i][j][k]][j][k] = src[i][j][k] (if dim=0)
    # Here dim=1: out[i][index[i][j][k]][k] = src[i][j][k]
    def torch_scatter(inputs, dim, index, src):
        # inputs shape: (2, 3, 4), dim: 1, index shape: (2, 2, 3), src shape: (2, 2, 3)
        
        # Create indices for the dimensions not being scattered
        # Dim 0 indices
        idx_0 = tf.range(tf.shape(inputs)[0])
        idx_0 = tf.reshape(idx_0, [tf.shape(inputs)[0], 1, 1])
        idx_0 = tf.broadcast_to(idx_0, tf.shape(index))
        
        # Dim 2 indices (last dimension)
        # The size of the last dimension of src determines how many indices we need for the last dim of inputs
        idx_2 = tf.range(tf.shape(src)[2])
        idx_2 = tf.reshape(idx_2, [1, 1, tf.shape(src)[2]])
        idx_2 = tf.broadcast_to(idx_2, tf.shape(index))
        
        # Stack indices to form (batch, index, depth) coordinates
        # index tensor provides the indices for dimension 1
        full_indices = tf.stack([idx_0, index, idx_2], axis=-1)
        full_indices = tf.reshape(full_indices, [-1, tf.rank(inputs)])
        
        # Flatten src to match the number of updates
        updates = tf.reshape(src, [-1])
        
        return tf.tensor_scatter_nd_update(inputs, full_indices, updates)

    for i in range(1000):
        # torch.cuda.empty_cache()
        # TensorFlow manages memory automatically and does not have a direct equivalent 
        # that is typically called inside a training loop.
        
        with tf.device('/GPU:0'):  # Equivalent to .cuda()
            # Create tensors
            inputs = tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4])
            src = tf.reshape(tf.range(-99, -99 + (2 * 2 * 3), dtype=tf.float32), [2, 2, 3])
            index = tf.constant(
                [
                    [
                        [0, 1, 0],
                        [1, 1, 0],
                    ],
                    [
                        [1, 0, 1],
                        [0, 0, 1],
                    ],
                ],
                dtype=tf.int64,
            )

            # inputs.requires_grad = True, src.requires_grad = True
            # In TensorFlow, we use tf.Variable to track gradients
            inputs_var = tf.Variable(inputs)
            src_var = tf.Variable(src)

            with tf.GradientTape() as tape:
                # res = torch.scatter(inputs, 1, index, src)
                res = torch_scatter(inputs_var, 1, index, src_var)

            # res.backward(torch.ones_like(res))
            # In TensorFlow, tape.gradient computes gradients of the target (res) w.r.t sources.
            # By default, it computes the gradient of the sum of the target, which matches backward(ones).
            grads = tape.gradient(res, [inputs_var, src_var])

            print(f"Test {i + 1}/{1000}")
            np.testing.assert_allclose(res.numpy(), gt_res)
            np.testing.assert_allclose(grads[0].numpy(), gt_input_grad)
            np.testing.assert_allclose(grads[1].numpy(), gt_src_grad)


if __name__ == "__main__":
    test_non_deterministic()
```