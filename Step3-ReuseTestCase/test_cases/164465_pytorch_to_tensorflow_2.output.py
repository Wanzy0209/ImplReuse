import torch
import tensorflow as tf
from tensorflow.experimental import dtensor

def test_int64_arange_max_copy_to_mesh():
    """
    Adapted test case for tf.experimental.dtensor.copy_to_mesh based on 
    PyTorch issue 164465 (inductor crash with int64 + arange + max).
    
    This test verifies the behavior of the similar API when handling
    operations involving int64 ranges and reductions, followed by
    distribution to a mesh.
    """
    
    # Setup a simple mesh for DTensor
    # Using a single device mesh to ensure the test runs without a cluster
    mesh = dtensor.create_mesh([('batch', [1])], devices=['CPU:0'])
    layout = dtensor.Layout.replicated(mesh, rank=0)

    # Inputs
    # x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
    x = tf.ones((1, 64), dtype=tf.int64)
    # y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
    y = tf.random.normal((64, 3072), dtype=tf.bfloat16)

    # Reproduce the logic from the PyTorch function
    # embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1)
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
    # Equivalent in TF: tf.range
    iota = tf.range(0, 36, dtype=tf.int64)

    # view_3 = torch.ops.aten.view.default(iota, [1, 36])
    view_3 = tf.reshape(iota, [1, 36])

    # max_1 = torch.ops.aten.max.default(view_3)
    max_1 = tf.reduce_max(view_3)

    # Adaptation: Use the similar API copy_to_mesh
    # The original function returns (max_1,). We copy this result to the mesh.
    # This tests if the API handles the int64 scalar result correctly.
    result_dtensor = dtensor.copy_to_mesh(max_1, layout)

    # Verify the result
    # The max of range(0, 36) is 35.
    assert result_dtensor.numpy() == 35, f"Expected 35, got {result_dtensor.numpy()}"

if __name__ == "__main__":
    test_int64_arange_max_copy_to_mesh()
    print("Test passed successfully.")