# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

def example_function():

    def logp(x, matrix):
        # print(matrix.is_contiguous()) # Uncomment this line to make the code run
        p_mat_sqrt = torch.linalg.cholesky(matrix).contiguous()
        p_mat_sqrt_inv = p_mat_sqrt.inverse()
        val = torch.sum((p_mat_sqrt_inv @ x[0, :]) ** 2)
        return -val/2

    score_func = torch.vmap(torch.func.grad(logp, 0), (0, None))

    return score_func

if __name__ == "__main__":

    device = torch.device("mps")
    dtype = torch.float32
    data = torch.zeros((2, 5, 3), device=device, dtype=dtype)
    compiled_function = torch.compile(example_function())

    p = torch.diag(torch.tensor((20., 0.5, 5,), device=device, dtype=dtype)**2)
    res = compiled_function(data, p)