import torch
import unittest

class TestCompileUnsqueezeCatInterleaving(unittest.TestCase):
    def test_compile_unsqueeze_cat_crash(self):
        """
        Reproduces issue 164700 where torch.compile generates illegal triton 
        tl.broadcast_to and crashes from particular unsqueeze and torch.cat 
        interleaving.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

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

        # Inputs matching the bug report
        x = torch.zeros(1, 32, dtype=torch.int64, device=device)
        y = torch.zeros(1, dtype=torch.int32, device=device)

        # 1. Run eager mode to establish baseline
        expected = f(x, y)

        # 2. Run compiled mode - this is where the crash occurred
        compiled_f = torch.compile(f)
        actual = compiled_f(x, y)

        # 3. Verify correctness
        self.assertTrue(torch.equal(expected, actual))

if __name__ == "__main__":
    unittest.main()