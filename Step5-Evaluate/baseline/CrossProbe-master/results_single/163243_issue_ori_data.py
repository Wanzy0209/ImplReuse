# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch


def repro():
    n = 8

    dtype = torch.complex64

    A = torch.randn(4, n, n, dtype=dtype, requires_grad=True)
    A = A.clone(memory_format=torch.contiguous_format)

    I0 = torch.eye(n, dtype=A.dtype, device=A.device)
    I = I0.unsqueeze(0).expand(A.shape[0], n, n).contiguous()

    A = I + 0.5 * (A @ A.mH)

    R = torch.linalg.cholesky(A, upper=True)
    loss = R.abs().sum()
    loss.backward()


if __name__ == '__main__':
    repro = torch.compile(repro, backend="inductor")

    repro()