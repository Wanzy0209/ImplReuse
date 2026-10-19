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
        # torch.cuda.empty_cache() -> TensorFlow manages memory automatically
        # inputs = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4]).cuda()
        inputs = tf.Variable(tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4]))
        
        # src = torch.arange(-99, -99 + (2 * 2 * 3), dtype=torch.float32).reshape([2, 2, 3]).cuda()
        src = tf.Variable(tf.reshape(tf.range(-99, -99 + 12, dtype=tf.float32), [2, 2, 3]))
        
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

        # inputs.requires_grad = True, src.requires_grad = True -> Handled by tf.Variable
        # res = torch.scatter(inputs, 1, index, src)
        # TensorFlow equivalent: tf.tensor_scatter_nd_update
        # We need to construct the full indices for the scatter operation.
        # PyTorch scatter dim=1 means we are modifying dimension 1.
        # index shape is (2, 2, 3). inputs shape is (2, 3, 4).
        # We need to map (b, i, j) in index to (b, index[b,i,j], j) in inputs.
        
        with tf.GradientTape() as tape:
            # Construct coordinates for tf.tensor_scatter_nd_update
            # Shape of index: (2, 2, 3)
            # We need indices of shape (12, 3) for the scatter op
            
            # Dim 0 (Batch): 0,0,0... 1,1,1...
            batch_idx = tf.range(2, dtype=tf.int64)
            batch_idx = tf.reshape(batch_idx, [2, 1, 1])
            batch_idx = tf.broadcast_to(batch_idx, [2, 2, 3])
            
            # Dim 1 (Scatter Dim): The values from 'index' tensor
            scatter_idx = index
            
            # Dim 2 (Last Dim): 0,1,2... 0,1,2...
            last_dim_idx = tf.range(3, dtype=tf.int64)
            last_dim_idx = tf.reshape(last_dim_idx, [1, 1, 3])
            last_dim_idx = tf.broadcast_to(last_dim_idx, [2, 2, 3])
            
            # Stack them to get (2, 2, 3, 3) then reshape to (12, 3)
            indices = tf.stack([batch_idx, scatter_idx, last_dim_idx], axis=-1)
            indices = tf.reshape(indices, [-1, 3])
            
            # Flatten src to match the number of updates
            updates = tf.reshape(src, [-1])
            
            # Perform scatter
            res = tf.tensor_scatter_nd_update(inputs, indices, updates)

        # res.backward(torch.ones_like(res))
        grads = tape.gradient(res, [inputs, src])
        input_grad, src_grad = grads[0], grads[1]

        print(f"Test {i + 1}/{1000}")
        # np.testing.assert_allclose(res.cpu().detach().numpy(), gt_res)
        np.testing.assert_allclose(res.numpy(), gt_res)
        # np.testing.assert_allclose(inputs.grad.cpu().numpy(), gt_input_grad)
        np.testing.assert_allclose(input_grad.numpy(), gt_input_grad)
        # np.testing.assert_allclose(src.grad.cpu().numpy(), gt_src_grad)
        np.testing.assert_allclose(src_grad.numpy(), gt_src_grad)


if __name__ == "__main__":
    test_non_deterministic()
```