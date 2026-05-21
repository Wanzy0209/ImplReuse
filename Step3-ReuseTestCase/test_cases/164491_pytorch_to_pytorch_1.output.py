import torch
import unittest

class TestIstftNonContiguous(unittest.TestCase):
    def test_istft_non_contiguous_input(self):
        """
        Test that torch.istft handles non-contiguous (strided) inputs correctly.
        This is inspired by the issue where _scaled_mm and _int_mm were slow 
        or raised errors with row-major (non-contiguous) rhs matrices.
        """
        # Parameters for istft
        n_fft = 16
        hop_length = 4
        win_length = 16
        frames = 10
        batch = 2

        # Create a contiguous complex input tensor
        # Shape: (batch, n_fft // 2 + 1, frames)
        input_cont = torch.randn(batch, n_fft // 2 + 1, frames, dtype=torch.complex64)
        
        # Create a non-contiguous version by transposing dimensions.
        # We create a tensor with swapped dimensions, transpose it back to the target shape,
        # resulting in a non-contiguous tensor.
        # Shape: (batch, frames, n_fft // 2 + 1) -> transpose -> (batch, n_fft // 2 + 1, frames)
        input_non_cont = torch.randn(batch, frames, n_fft // 2 + 1, dtype=torch.complex64).transpose(1, 2)
        
        # Verify that the input is indeed non-contiguous
        self.assertFalse(input_non_cont.is_contiguous(), "Test input must be non-contiguous")

        window = torch.hann_window(window_length=win_length)

        # Run istft on contiguous input
        try:
            output_cont = torch.istft(
                input_cont, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window,
                return_complex=False
            )
        except Exception as e:
            self.fail(f"torch.istft failed with contiguous input: {e}")

        # Run istft on non-contiguous input
        try:
            output_non_cont = torch.istft(
                input_non_cont, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window,
                return_complex=False
            )
        except Exception as e:
            self.fail(f"torch.istft failed with non-contiguous input: {e}")

        # Verify that outputs are numerically close
        self.assertTrue(torch.allclose(output_cont, output_non_cont, atol=1e-5))

if __name__ == '__main__':
    unittest.main()