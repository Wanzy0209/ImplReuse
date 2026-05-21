import torch


def repro():
    n = 8

    dtype = torch.complex64

    A = torch.randn(4, n, n, dtype=dtype, requires_grad=True)
    A = A.clone(memory_format=torch.contiguous_format)

    # Leverage the similar API: torch.eq
    # We use torch.eq to compare the tensor with its conjugate transpose (.mH).
    # This integrates the equality check logic into the inductor graph
    # alongside the problematic .mH view operation.
    hermitian_check = torch.eq(A, A.mH)

    I0 = torch.eye(n, dtype=A.dtype, device=A.device)
    I = I0.unsqueeze(0).expand(A.shape[0], n, n).contiguous()

    # Original problematic logic involving .mH
    A = I + 0.5 * (A @ A.mH)

    R = torch.linalg.cholesky(A, upper=True)
    loss = R.abs().sum()
    loss.backward()


if __name__ == '__main__':
    # Compile with the backend that triggers the bug
    repro = torch.compile(repro, backend="inductor")

    repro()