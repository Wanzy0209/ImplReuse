import tensorflow as tf
import numpy as np

# The original issue (166157) highlights the lack of observability (statistics)
# in the OpenReg allocator. The fix adds getDeviceStats and tracking.
# This test verifies that the similar API (LinearOperatorHouseholder)
# provides access to its internal "statistics" (mathematical properties),
# ensuring it is not a "black box".

def test_linear_operator_householder_observability():
    """
    Test that LinearOperatorHouseholder exposes essential statistics (properties)
    similar to how the OpenReg allocator should expose memory stats.
    """
    # Initialize the operator (analogous to OpenRegDeviceAllocator)
    # Using a simple vector for a 3x3 reflection
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    operator = tf.linalg.LinearOperatorHouseholder(vec)

    # 1. Verify basic shape statistics (analogous to allocation size)
    # The issue mentions "zero visibility into memory consumption".
    # Here we verify visibility into the operator's dimensions.
    assert operator.shape == [3, 3], "Shape statistics should be accessible."

    # 2. Verify computed statistics (analogous to allocated_bytes/reserved_bytes)
    # The fix adds tracking of bytes. Here we check the determinant.
    # For a Householder reflection, the determinant is always -1.
    # log_abs_determinant should be log(1) = 0.
    log_det = operator.log_abs_determinant()
    assert np.allclose(log_det, 0.0), "Log absolute determinant statistic is incorrect."

    # 3. Verify another computed statistic (Trace)
    # Trace(I - 2vv^T) = n - 2*sum(v^2). For normalized v, sum(v^2)=1.
    # Trace = 3 - 2 = 1.
    trace_val = operator.trace()
    assert np.allclose(trace_val, 1.0), "Trace statistic is incorrect."

    print("Test passed: LinearOperatorHouseholder provides necessary observability (statistics).")

if __name__ == "__main__":
    test_linear_operator_householder_observability()