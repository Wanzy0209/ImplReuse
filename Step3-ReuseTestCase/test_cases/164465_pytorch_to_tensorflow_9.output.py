import torch
import tensorflow as tf

def test_tf_name_scope_int64_arange_max():
    """
    Adapted from PyTorch bug report (Issue ID: 164465).
    Original issue: Inductor crash with int64 + arange + max inside torch.compile.
    
    This test adapts the logic to TensorFlow, wrapping the operations in 
    tf.name_scope (the identified similar API) to verify behavior.
    """
    
    # Check for GPU availability to mimic the original 'cuda' device requirement
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/GPU:0' if gpus else '/CPU:0'
    
    with tf.device(device_name):
        # Define inputs
        # x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
        x = tf.ones((1, 64), dtype=tf.int64)
        
        # y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
        y = tf.random.normal((64, 3072), dtype=tf.bfloat16)

        # Wrap logic in the similar API: tf.name_scope
        with tf.name_scope("bug_repro_scope"):
            # embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
            # PyTorch embedding: (weights, indices). TF gather: (params, indices).
            embedding = tf.gather(y, x)

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

            # iota = torch.ops.prims.iota.default(36, start = 0, step = 1, dtype = torch.int64, device = 'cuda', requires_grad = False)
            iota = tf.range(0, 36, dtype=tf.int64)

            # view_3 = torch.ops.aten.view.default(iota, [1, 36])
            view_3 = tf.reshape(iota, [1, 36])

            # max_1 = torch.ops.aten.max.default(view_3)
            max_1 = tf.reduce_max(view_3)

    # Verify the result
    # The iota creates a range [0, 35], so the max should be 35.
    expected_max = 35
    assert max_1.numpy() == expected_max, f"Expected max to be {expected_max}, got {max_1.numpy()}"
    
    print("Test passed. tf.name_scope executed successfully with int64 + range + max logic.")

if __name__ == "__main__":
    test_tf_name_scope_int64_arange_max()