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
    for i in range(1000):
        # torch.cuda.empty_cache() # TF manages memory automatically
        
        # Create tensors
        # inputs = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4]).cuda()
        inputs = tf.Variable(tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4]), trainable=True)
        
        # src = torch.arange(-99, -99 + (2 * 2 * 3), dtype=torch.float32).reshape([2, 2, 3]).cuda()
        src = tf.Variable(tf.reshape(tf.range(-99, -99 + (2 * 2 * 3), dtype=tf.float32), [2, 2, 3]), trainable=True)
        
        # index = torch.tensor(..., dtype=torch.int64).cuda()
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

        # inputs.requires_grad = True # Handled by tf.Variable(trainable=True)
        # src.requires_grad = True

        with tf.GradientTape() as tape:
            # PyTorch: res = torch.scatter(inputs, 1, index, src)
            # TensorFlow equivalent: tf.tensor_scatter_nd_update
            # We need to construct the full indices for the scatter operation.
            # PyTorch scatter logic for dim=1: input[i][index[i][j][k]][k] = src[i][j][k]
            
            # Create indices for dimension 0 (batch)
            i_indices = tf.range(2, dtype=tf.int64)
            i_indices = tf.reshape(i_indices, [2, 1, 1])
            i_indices = tf.broadcast_to(i_indices, [2, 2, 3])

            # Create indices for dimension 2 (column/depth)
            k_indices = tf.range(3, dtype=tf.int64)
            k_indices = tf.reshape(k_indices, [1, 1, 3])
            k_indices = tf.broadcast_to(k_indices, [2, 2, 3])

            # Stack indices to get shape (2, 2, 3, 3) -> (batch, row, col, coord)
            indices = tf.stack([i_indices, index, k_indices], axis=-1)
            # Reshape to (N, 3) for scatter_nd_update
            indices = tf.reshape(indices, [-1, 3])

            # Flatten src to (N,)
            updates = tf.reshape(src, [-1])

            res = tf.tensor_scatter_nd_update(inputs, indices, updates)

            # res.backward(torch.ones_like(res))
            # Equivalent to computing gradients of sum(res)
            loss = tf.reduce_sum(res)

        # Compute gradients
        grads = tape.gradient(loss, [inputs, src])
        inputs_grad, src_grad = grads

        print(f"Test {i + 1}/{1000}")
        np.testing.assert_allclose(res.numpy(), gt_res)
        np.testing.assert_allclose(inputs_grad.numpy(), gt_input_grad)
        # Note: PyTorch scatter gradient behavior with duplicate indices can be undefined/implementation specific.
        # TensorFlow's tensor_scatter_nd_update might differ if indices collide.
        np.testing.assert_allclose(src_grad.numpy(), gt_src_grad)


if __name__ == "__main__":
    test_non_deterministic()
```