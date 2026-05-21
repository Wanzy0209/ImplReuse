import torch
import tensorflow as tf
from tf.experimental import dtensor
import pytest

def test_relayout_shape_constraint_violation():
    """
    Adapted from PyTorch test_dynamic_cache_exportability_multiple_run.
    
    Original Bug: torch.export.export fails with UserError: Constraints violated (seq).
    The error occurred because the dynamic shape constraints (seq <= 512) were violated
    during the export process.
    
    This test verifies that the TensorFlow equivalent, tf.experimental.dtensor.relayout,
    correctly handles constraint violations. In dtensor, a primary constraint is that
    a tensor's dimension size must be divisible by the mesh size if it is to be sharded
    along that dimension.
    """
    # Initialize a mesh with 2 devices in 'x' dimension.
    # This imposes a constraint: any dimension sharded on 'x' must be divisible by 2.
    mesh = dtensor.Mesh(['x', 'y'], [2, 2])

    # Case 1: Valid Relayout (Constraint Satisfied)
    # Shape (4, 4) is divisible by mesh size (2, 2).
    t1 = tf.zeros((4, 4), dtype=tf.float32)
    
    # Start with a replicated layout
    layout_replicated = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)
    dt1 = dtensor.DTensor(t1, layout=layout_replicated)

    # Target layout: Shard on 'x'
    layout_sharded_x = dtensor.Layout([dtensor.SHARDED, dtensor.UNSHARDED], mesh)

    # This should succeed as the constraint (4 % 2 == 0) is met.
    dt1_relayouted = dtensor.relayout(dt1, layout_sharded_x)
    assert dt1_relayouted.layout == layout_sharded_x

    # Case 2: Invalid Relayout (Constraint Violation)
    # Shape (3, 4) is NOT divisible by mesh dimension 'x' (size 2).
    # This mimics the "Constraints violated" error from the PyTorch issue.
    t2 = tf.zeros((3, 4), dtype=tf.float32)
    dt2 = dtensor.DTensor(t2, layout=layout_replicated)

    # Attempting to shard dimension 0 (size 3) on mesh 'x' (size 2)
    # should raise a ValueError or InvalidArgumentError due to constraint violation.
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        dtensor.relayout(dt2, layout_sharded_x)