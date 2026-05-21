import torch
import tensorflow as tf
from tensorflow.python.ops import control_flow_util

# Reproduce the structure of the original bug report's 'inner_f' class
class InnerF(tf.Module):
    def __init__(self):
        super().__init__()
        # Mimicking constants or state if necessary
        self._tensor_constant0 = tf.constant([1, 2, 3])

    # Applying the similar API (tf.compat.v1.enable_control_flow_v2)
    # This acts as the wrapper/transformer analogous to aot_export in the bug.
    @tf.compat.v1.enable_control_flow_v2
    def forward(self, primals, tangents):
        # The bug report highlights incorrect stack traces/annotations.
        # Here we verify the 'annotation' (control flow version) is correct.
        # The implementation of enable_control_flow_v2 sets this flag to True.
        assert control_flow_util.ENABLE_CONTROL_FLOW_V2, \
            "Control flow V2 should be enabled inside the decorated method"

        # Mimic the operations in the original bug report (simplified for TF)
        # Original: t = torch.ops.aten.t.default(primals_1)
        # TF equivalent:
        t = tf.transpose(primals)

        # Original: mm = torch.ops.aten.mm.default(primals_2, t)
        # TF equivalent:
        mm = tf.matmul(tangents, t)

        # Return a result
        return mm

def test_enable_control_flow_v2_metadata_integrity():
    """
    Test case to verify that tf.compat.v1.enable_control_flow_v2 correctly
    manages the control flow state (metadata), analogous to how the bug report
    expects stack traces to be preserved (but they were wrong).
    """
    # Save original state
    original_state = control_flow_util.ENABLE_CONTROL_FLOW_V2

    # Ensure we start in a known state (False)
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = False

    module = InnerF()

    # Define inputs
    primals = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tangents = tf.constant([[1.0, 0.0], [0.0, 1.0]])

    # Execute the decorated method
    # In the bug, this execution (via aot_export) resulted in wrong metadata.
    # Here we execute and check the metadata (flag) behavior.
    result = module.forward(primals, tangents)

    # Verify the result is computed (sanity check)
    assert result.shape == (2, 2)

    # Verify the state is restored after execution.
    # The bug report shows metadata being wrong. We assert it is correct here.
    assert control_flow_util.ENABLE_CONTROL_FLOW_V2 == False, \
        "Control flow V2 flag should be reset to False after execution"

    # Restore original state
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = original_state

if __name__ == "__main__":
    test_enable_control_flow_v2_metadata_integrity()
    print("Test passed.")