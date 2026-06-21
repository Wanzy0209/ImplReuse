import numpy as np
import torch


def test_cholesky_inverse_determinism():
    # Define a valid Cholesky factor (Lower triangular)
    # Input matrix L
    input_data = np.array(
        [
            [4.0, 0.0, 0.0],
            [2.0, 5.0, 0.0],
            [1.0, -1.0, 3.0],
        ],
        dtype=np.float32,
    )

    # Calculate ground truth result and gradient once using PyTorch on CPU
    # to serve as the reference for the loop.
    ref_input = torch.tensor(input_data, requires_grad=True)
    ref_res = torch.cholesky_inverse(ref_input, upper=False)
    ref_res.backward(torch.ones_like(ref_res))

    gt_res = ref_res.detach().numpy()
    gt_grad = ref_input.grad.detach().numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(1000):
        if device == "cuda":
            torch.cuda.empty_cache()

        # Move to device
        x = torch.tensor(input_data, device=device, dtype=torch.float32)
        x.requires_grad = True

        # Call the similar API: torch.cholesky_inverse
        res = torch.cholesky_inverse(x, upper=False)

        # Backward pass
        res.backward(torch.ones_like(res))

        print(f"Test {i + 1}/1000")

        # Assert determinism and correctness against reference
        np.testing.assert_allclose(res.cpu().detach().numpy(), gt_res, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(x.grad.cpu().numpy(), gt_grad, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_cholesky_inverse_determinism()