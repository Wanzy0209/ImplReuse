import torch
import tensorflow as tf

def test_int64_arange_max_in_name_scope():
    """
    Adapted test case from PyTorch issue 164465.
    Verifies the behavior of int64 arange (iota) and max operations
    within a tf.keras.name_scope, mirroring the logic of the original
    torch.compile crash scenario.
    """
    
    # Setup inputs matching the PyTorch repro
    # x: indices (int64)
    # y: weights (bfloat16)
    x = tf.ones((1, 64), dtype=tf.int64)
    y = tf.random.normal((64, 3072), dtype=tf.bfloat16)

    # Use the requested similar API: tf.keras.name_scope
    with tf.keras.name_scope("inductor_crash_repro"):
        # embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
        # PyTorch: (weights, indices) -> TF: (params, ids)
        embedding = tf.nn.embedding_lookup(y, x)

        # view = torch.ops.aten.view.default(embedding, [64, 3072])
        view = tf.reshape(embedding, [64, 3072])

        # unsqueeze = torch.ops.aten.unsqueeze.default(view, 0)
        unsqueeze = tf.expand_dims(view, 0)

        # expand = torch.ops.aten.expand.default(unsqueeze, [576, -1, -1])
        expand = tf.broadcast_to(unsqueeze, [576, -1, -1])

        # view_1 = torch.ops.aten.view.default(expand, [2, 8, 36, 64, 3072])
        view_1 = tf.reshape(expand, [2, 8, 36, 64, 3072])

        # permute = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2, 4])
        permute = tf.transpose(view_1, [0, 1, 3, 2, 4])

        # clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        clone = tf.identity(permute)

        # view_2 = torch.ops.aten.view.default(clone, [2, 18432, 3072])
        view_2 = tf.reshape(clone, [2, 18432, 3072])

        # iota = torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, ...)
        iota = tf.range(0, 36, dtype=tf.int64)

        # view_3 = torch.ops.aten.view.default(iota, [1, 36])
        view_3 = tf.reshape(iota, [1, 36])

        # max_1 = torch.ops.aten.max.default(view_3)
        # In PyTorch, max() on a tensor returns the single maximum value.
        max_val = tf.reduce_max(view_3)

    # Verify the result
    # The iota creates [0, 1, ..., 35], so the max should be 35.
    expected_max = 35
    
    # Assert to ensure the logic executed correctly
    assert max_val.numpy() == expected_max, f"Expected max to be {expected_max}, got {max_val.numpy()}"
    
    print("Test passed. tf.keras.name_scope handled the operations correctly.")

if __name__ == "__main__":
    test_int64_arange_max_in_name_scope()