import torch
import unittest
import os

class TestTorchCompileIssue164700(unittest.TestCase):
    def test_cat_unsqueeze_interleaving(self):
        """
        Test for Issue 164700: torch.compile generates illegal triton tl.broadcast_to
        and crashes from particular unsqueeze and torch.cat interleaving.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Setting logs as in the original bug report to help debugging if needed
        os.environ["TORCH_LOGS"] = "output_code"

        device = "cuda"

        def f(x, y):
            y2 = torch.cat(
                [
                    x[:, 1:],
                    y[:, None] + 32 * 2048,
                ],
                dim=1,
            )

            x2 = x[:, 1:, None]
            y3 = y2[:, -1:, None]

            return (
                torch.cat([x2, y3], dim=1)
                + torch.arange(-2048, 0, device=device)[None, None, :]
            ).reshape(1, 32 * 2048)

        # Note: The original bug report used int32 for y, but torch.cat requires matching dtypes.
        # We use int64 here to ensure the eager execution succeeds as claimed in the bug report.
        x = torch.zeros(1, 32, dtype=torch.int64, device=device)
        y = torch.zeros(1, dtype=torch.int64, device=device)

        # 1. Test eager execution (This should succeed)
        expected = f(x, y)
        self.assertIsNotNone(expected)

        # 2. Test compiled execution (This was crashing)
        compiled_f = torch.compile(f)
        result = compiled_f(x, y)

        # 3. Verify results match
        self.assertTrue(torch.equal(expected, result))

if __name__ == "__main__":
    unittest.main()