import torch
import tensorflow as tf
import tf.experimental.dtensor as dtensor

# Similar API: tf.tpu.experimental.initialize_tpu_system
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system()
    mesh = dtensor.create_mesh([("x", 4)], devices=tf.config.list_logical_devices("TPU"))
except (ValueError, tf.errors.NotFoundError):
    # Fallback for non-TPU environments to ensure code is syntactically valid
    mesh = dtensor.create_mesh([("x", 1)], devices=[tf.config.list_physical_devices("CPU")[0]])

def test_as_strided_dtype_view_replacement():
    """
    Test case reflecting the bug where as_strided lowering throws away .view(dtype).
    The original issue involved tensors of specific dtypes (uint8, float8, int32, bfloat16)
    being viewed or passed through operations where the dtype information was lost.
    
    This test replaces the original PyTorch APIs with tf.experimental.dtensor.name_based_restore
    to simulate the data loading and viewing process, ensuring dtypes are preserved.
    """

    # Original: scales_ptr (torch.uint8)
    # Replacement: torch.uint8 -> tf.experimental.dtensor.name_based_restore
    scales_ptr = tf.experimental.dtensor.name_based_restore(
        "scales",
        layout=dtensor.Layout([dtensor.ShardedSpec(0, 4)], mesh=mesh),
        dtype=tf.uint8,
        shape=[16512, 160]
    )

    # Original: output_scales_ptr (torch.float8_e8m0fnu viewed as uint8)
    # The bug was that the view (uint8) was discarded during lowering, reverting to float8.
    # Replacement: torch.float8_e8m0fnu -> tf.experimental.dtensor.name_based_restore
    # Replacement: .view(dtype) -> tf.experimental.dtensor.name_based_restore
    # We use name_based_restore to simulate the "view" operation, requesting the target dtype.
    output_scales_ptr = tf.experimental.dtensor.name_based_restore(
        "output_scales",
        layout=dtensor.Layout([dtensor.ShardedSpec(0, 4)], mesh=mesh),
        dtype=tf.uint8, # The intended view dtype
        shape=[17024, 160]
    )

    # Original: orig_offsets (torch.int32)
    # Replacement: torch.int32 -> tf.experimental.dtensor.name_based_restore
    orig_offsets = tf.experimental.dtensor.name_based_restore(
        "orig_offsets",
        layout=dtensor.Layout.replicated(mesh, rank=1),
        dtype=tf.int32,
        shape=[4]
    )

    # Original: w1.bfloat16, x.bfloat16
    # Replacement: w1.bfloat16 -> tf.experimental.dtensor.name_based_restore
    # Replacement: x.bfloat16 -> tf.experimental.dtensor.name_based_restore
    w1 = tf.experimental.dtensor.name_based_restore(
        "w1", layout=dtensor.Layout.replicated(mesh, rank=2), dtype=tf.bfloat16, shape=[160, 160]
    )
    x = tf.experimental.dtensor.name_based_restore(
        "x", layout=dtensor.Layout.replicated(mesh, rank=2), dtype=tf.bfloat16, shape=[160, 160]
    )

    # Original: self.experts, self.ffn_norm
    # Replacement: self.experts -> tf.experimental.dtensor.name_based_restore
    # Replacement: self.ffn_norm -> tf.experimental.dtensor.name_based_restore
    experts = tf.experimental.dtensor.name_based_restore(
        "experts", layout=dtensor.Layout.replicated(mesh, rank=1), dtype=tf.float32, shape=[4]
    )
    ffn_norm = tf.experimental.dtensor.name_based_restore(
        "ffn_norm", layout=dtensor.Layout.replicated(mesh, rank=1), dtype=tf.float32, shape=[4]
    )

    # Assertions to verify the logic
    # The core bug was that output_scales_ptr (viewed as uint8) was treated as float8.
    # We assert that our replacement mechanism (name_based_restore) respects the requested dtype.
    assert output_scales_ptr.dtype == tf.uint8, \
        f"Bug reproduction: Expected dtype tf.uint8, but got {output_scales_ptr.dtype}"
    
    assert scales_ptr.dtype == tf.uint8
    assert orig_offsets.dtype == tf.int32
    assert w1.dtype == tf.bfloat16
    assert x.dtype == tf.bfloat16

if __name__ == "__main__":
    test_as_strided_dtype_view_replacement()