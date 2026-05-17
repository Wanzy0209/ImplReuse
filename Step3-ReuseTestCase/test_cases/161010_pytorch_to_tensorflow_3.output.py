import torch
import tensorflow as tf

# The API under test: enable_eager_execution
# This ensures the operations run immediately, similar to PyTorch's default eager mode.
tf.compat.v1.enable_eager_execution()

def test_eager_stride_preservation():
    # Setup input data
    # PyTorch: A = torch.rand(5, 5, device="cuda" if torch.cuda.is_available() else "cpu")
    A = tf.random.uniform((5, 5))

    def f(A, count):
        # PyTorch: Q, R = torch.linalg.qr(A)
        Q, R = tf.linalg.qr(A)

        # PyTorch: rhs = torch.ones(Q.shape[0], 1, device=A.device)
        rhs = tf.ones([tf.shape(Q)[0], 1], dtype=A.dtype)

        # PyTorch: a = torch.linalg.solve_triangular(R, Q.T @ rhs, upper=True)
        # TensorFlow: tf.linalg.triangular_solve(matrix, rhs, lower=False)
        # Note: PyTorch's upper=True corresponds to TensorFlow's lower=False.
        matmul_res = tf.linalg.matmul(tf.transpose(Q), rhs)
        a = tf.linalg.triangular_solve(R, matmul_res, lower=False)

        # PyTorch: if a.stride() == a.clone(memory_format=torch.preserve_format).stride():
        # TensorFlow: Check if strides are preserved through an identity operation.
        # Note: TF strides are in bytes, PyTorch strides are in elements.
        # The logic checks for equality of the stride tuple structure.
        if a.strides == tf.identity(a).strides:
            return count + 1
        return count

    # Initialize count
    # PyTorch: torch.zeros(1)
    count = tf.zeros(1, dtype=tf.float32)

    # Run the function in eager mode
    res = f(A, count)
    print(f"Eager Result: {res}")

    # Verify the result
    # In the original PyTorch bug report, eager mode returns 1.0 (count + 1) because strides match.
    # We expect the same correct behavior here with enable_eager_execution.
    assert res.numpy()[0] == 1.0, "Eager execution should preserve stride layout"

if __name__ == "__main__":
    test_eager_stride_preservation()