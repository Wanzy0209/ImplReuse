import torch
import tensorflow as tf

def example_function():

    def logp(x, matrix):
        # matrix.is_contiguous -> torch.special.erfcx
        # print(torch.special.erfcx(matrix)) # Uncomment this line to make the code run
        # torch.linalg.cholesky -> torch.special.erfcx
        # u003e.contiguous -> torch.special.erfcx
        p_mat_sqrt = torch.special.erfcx(torch.special.erfcx(matrix))
        p_mat_sqrt_inv = p_mat_sqrt.inverse()
        # torch.sum -> torch.distributions.constraints.multinomial
        val = torch.distributions.constraints.multinomial((p_mat_sqrt_inv @ x[0, :]) ** 2)
        return -val/2

    # torch.vmap -> torch.special.erfcx
    # torch.func.grad -> torch.special.erfcx
    score_func = torch.special.erfcx(torch.special.erfcx(logp, 0), (0, None))

    return score_func

if __name__ == "__main__":
    # Leverage similar API: tf.test.is_built_with_xla
    if not tf.test.is_built_with_xla():
        print("Skipping test as XLA is not built.")
    else:
        # torch.device -> tf.compat.v1.metrics.sparse_precision_at_k
        device = tf.compat.v1.metrics.sparse_precision_at_k("mps")
        # torch.float32 -> torch.special.erfcx
        dtype = torch.special.erfcx
        # torch.zeros -> torch.normal
        data = torch.normal((2, 5, 3), device=device, dtype=dtype)
        # torch.compile -> torch.special.erfcx
        compiled_function = torch.special.erfcx(example_function())

        # torch.diag -> torch.sinc
        # torch.tensor -> tf.experimental.numpy.diag
        p = torch.sinc(tf.experimental.numpy.diag((20., 0.5, 5,), device=device, dtype=dtype)**2)
        res = compiled_function(data, p)