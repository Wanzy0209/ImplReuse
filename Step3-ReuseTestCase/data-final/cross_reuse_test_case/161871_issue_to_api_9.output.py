import torch
import unittest

class TestMvlgammaFPE(unittest.TestCase):
    def test_mvlgamma_large_p_int64(self):
        """
        Test case for Issue 161871: Floating point exception in torch.Tensor.mvlgamma_
        
        The bug occurs when calling mvlgamma_ with a large integer p (1024) 
        and an int64 tensor, causing a Floating Point Exception (core dump).
        This test verifies that the operation completes without crashing the interpreter,
        either by returning NaNs/Infs (mathematically expected for these inputs) 
        or raising a Python exception.
        """
        # Reproduce the exact setup from the bug report
        tensor = torch.randint(low=0, high=10, size=(5,), dtype=torch.int64)
        
        # Setup arguments using the unpacking pattern from the issue
        input_args = [tensor, 1024]
        input_kwargs = {}
        
        # The operation should not cause a core dump (Floating Point Exception).
        # We wrap it in a try-except block to catch Python-level errors if the fix
        # involves validation, but the primary goal is to ensure the process survives.
        try:
            torch.Tensor.mvlgamma_(*input_args, **input_kwargs)
        except (RuntimeError, ValueError) as e:
            # If the fix raises an error for invalid inputs (like p being too large for the data type),
            # that is an acceptable resolution to the crash.
            pass
        except FloatingPointError:
            # A Python FloatingPointError is better than a SIGFPE core dump, 
            # but we still want to note it.
            pass
            
        # If we reach here, the interpreter did not crash.
        # With p=1024 and values 0-9, the arguments to the log-gamma function 
        # will be negative integers, resulting in NaNs.
        # We assert that the tensor is modified (in-place) and contains NaNs or Infs,
        # confirming the operation was attempted and handled gracefully.
        self.assertTrue(torch.isnan(tensor).all() or torch.isinf(tensor).any())

if __name__ == '__main__':
    unittest.main()