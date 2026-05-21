import torch
import unittest

class TestMPSClamp(unittest.TestCase):
    def test_clamp_mps_backend(self):
        """
        Test case for Issue 167767: clamp incorrectness with mps backend.
        Leverages the backend availability check pattern similar to 
        torch.backends.mkldnn.is_available to ensure the test runs only on supported hardware.
        """
        # Reusing the pattern of checking backend availability
        if not torch.backends.mps.is_available():
            self.skipTest("MPS backend is not available")

        # The following line triggers the incorrect behavior in the original bug report.
        # It must be included to reproduce the state-dependent issue.
        a = torch.zeros(1, device='mps')
        a_clamped = a.clamp(min=0.0)

        # Test 1: Basic clamp with min
        b = torch.zeros(1, device='mps')
        c = b.clamp(min=1e-7)
        self.assertEqual(c[0], 1e-7, "Clamp with min=1e-7 failed on MPS backend")

        # Test 2: Clamp with min and explicit max=None
        b = torch.zeros(1, device='mps')
        c = b.clamp(min=1e-7, max=None)
        self.assertEqual(c[0], 1e-7, "Clamp with min=1e-7, max=None failed on MPS backend")

        # Test 3: Clamp with min and max=inf
        b = torch.zeros(1, device='mps')
        c = b.clamp(min=1e-7, max=torch.inf)
        self.assertEqual(c[0], 1e-7, "Clamp with min=1e-7, max=inf failed on MPS backend")

        # Test 4: Using clamp_min
        b = torch.zeros(1, device='mps')
        c = b.clamp_min(1e-7)
        self.assertEqual(c[0], 1e-7, "clamp_min(1e-7) failed on MPS backend")

if __name__ == '__main__':
    unittest.main()