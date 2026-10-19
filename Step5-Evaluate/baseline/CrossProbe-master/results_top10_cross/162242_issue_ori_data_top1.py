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
        # TensorFlow manages memory automatically; no direct equivalent to empty_cache needed
        # torch.cuda.empty_cache()
        
        # Conversion: torch.arange -> tf.range
        # Conversion: .reshape -> tf.reshape
        # Conversion: .cuda() -> TF uses GPU by default if available
        inputs = tf.range(24, dtype=tf.float32)
        inputs = tf.reshape(inputs, [2, 3, 4])
        
        src = tf.range(-99, -99 + (2 * 2 * 3), dtype=tf.float32)
        src = tf.reshape(src, [2, 2, 3])
        
        # Conversion: torch.tensor -> tf.constant
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

        # Conversion: requires_grad=True -> tf.GradientTape().watch()
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            tape.watch(src)

            # Conversion: torch.scatter(inputs, 1, index, src)
            # PyTorch scatter with dim=1: out[i][index[i][j][k]][k] = src[i][j][k]
            # TensorFlow equivalent: tf.tensor_scatter_nd_update
            # We need to construct absolute indices for TF from the relative indices of PyTorch
            
            batch_size = 2
            num_updates = 2
            width_dim = 3 # Corresponds to the 3rd dimension of index/src
            
            # Create batch indices [0, 0, ...], [1, 1, ...]
            batch_indices = tf.range(batch_size, dtype=tf.int64)
            batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1])
            batch_indices = tf.broadcast_to(batch_indices, [batch_size, num_updates, width_dim])
            
            # Create width indices [0, 1, 2], [0, 1, 2] ...
            width_indices = tf.range(width_dim, dtype=tf.int64)
            width_indices = tf.reshape(width_indices, [1, 1, width_dim])
            width_indices = tf.broadcast_to(width_indices, [batch_size, num_updates, width_dim])
            
            # Stack to get [batch, index, width] coordinates
            # index tensor is already [batch, num_updates, width]
            indices = tf.stack([batch_indices, index, width_indices], axis=-1)
            indices = tf.reshape(indices, [-1, 3])
            
            updates = tf.reshape(src, [-1])
            
            res = tf.tensor_scatter_nd_update(inputs, indices, updates)

        # Conversion: res.backward(torch.ones_like(res))
        # In TF, tape.gradient computes the sum of gradients by default, which is equivalent to passing ones_like
        grads = tape.gradient(res, [inputs, src])

        print(f"Test {i + 1}/{1000}")
        # Conversion: .cpu().detach().numpy() -> .numpy()
        np.testing.assert_allclose(res.numpy(), gt_res)
        np.testing.assert_allclose(grads[0].numpy(), gt_input_grad)
        np.testing.assert_allclose(grads[1].numpy(), gt_src_grad)


if __name__ == "__main__":
    test_non_deterministic()
```