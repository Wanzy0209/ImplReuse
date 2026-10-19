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

    # Determine device to use (GPU if available, mimicking .cuda())
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'

    for i in range(1000):
        # torch.cuda.empty_cache() is not required in TensorFlow
        # TF manages memory automatically
        
        with tf.device(device_name):
            # Conversion: torch.arange(...).cuda() -> tf.range(...) on device
            inputs = tf.Variable(tf.reshape(tf.range(24, dtype=tf.float32), [2, 3, 4]))
            
            # Conversion: torch.arange(...).cuda() -> tf.range(...) on device
            src = tf.Variable(tf.reshape(tf.range(-99, -99 + (2 * 2 * 3), dtype=tf.float32), [2, 2, 3]))
            
            # Conversion: torch.tensor(...).cuda() -> tf.constant(...) on device
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

            # Conversion: requires_grad=True is default for tf.Variable in GradientTape
            with tf.GradientTape(persistent=True) as tape:
                # Conversion: torch.scatter(inputs, 1, index, src)
                # TF equivalent: tf.tensor_scatter_nd_update
                # We need to construct the full indices for TF scatter operation
                # PyTorch scatter logic: inputs[b, index[b, i, j], j] = src[b, i, j]
                
                # Create coordinate grids
                b = tf.range(2, dtype=tf.int64)
                i_dim = tf.range(2, dtype=tf.int64)
                j_dim = tf.range(3, dtype=tf.int64)
                
                B, I, J = tf.meshgrid(b, i_dim, j_dim, indexing='ij')
                
                # Stack coordinates to form (batch, index, column) tuples
                # Shape: (2, 2, 3, 3)
                scatter_indices = tf.stack([B, index, J], axis=-1)
                
                # Flatten indices and updates for tf.tensor_scatter_nd_update
                flat_indices = tf.reshape(scatter_indices, [-1, 3])
                flat_updates = tf.reshape(src, [-1])
                
                res = tf.tensor_scatter_nd_update(inputs, flat_indices, flat_updates)

                # Conversion: res.backward(torch.ones_like(res))
                # In TF, we compute gradients of a loss. 
                # backward(ones_like) is equivalent to reducing sum of the output.
                loss = tf.reduce_sum(res)

            # Conversion: inputs.grad -> tape.gradient(loss, inputs)
            grad_inputs = tape.gradient(loss, inputs)
            # Conversion: src.grad -> tape.gradient(loss, src)
            grad_src = tape.gradient(loss, src)

            print(f"Test {i + 1}/{1000}")
            np.testing.assert_allclose(res.numpy(), gt_res)
            np.testing.assert_allclose(grad_inputs.numpy(), gt_input_grad)
            np.testing.assert_allclose(grad_src.numpy(), gt_src_grad)


if __name__ == "__main__":
    test_non_deterministic()
```