import torch
import sympy
from torch.testing._internal.common_utils import TestCase

class TestFloorDivSymbolicSimplification(TestCase):
    """
    Test case for FloorDiv generating sympy rational.
    This test reproduces the logic from the bug report where FloorDiv was 
    incorrectly simplified to Mul(Rational(...)) instead of maintaining 
    integer floor division semantics.
    """

    def test_floor_div_symbolic_expression(self):
        """
        Test that torch.div with floor rounding mode handles complex symbolic
        expressions correctly without simplifying to floating point rationals.
        """
        # The bug report involves symbolic variables s14, s37, s46.
        # We simulate this by using torch.compile with dynamic shapes or 
        # by checking the behavior of the operation directly.
        # Since the bug is in the symbolic lowering (inductor), we use torch.compile.

        # Define the function corresponding to the bug report expression:
        # FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
        def floor_div_expr(s14, s37, s46):
            # s14, s37, s46 are scalar tensors here
            inner_div = torch.div(s14 * s46, 2016, rounding_mode='floor')
            middle_term = (24 * s37 + 672) * inner_div
            numerator = middle_term + 21
            result = torch.div(numerator, 22, rounding_mode='floor')
            return result

        # Test with concrete values that match the symbolic constraints (positive integers)
        # s14=4032 (multiple of 2016), s37=1, s46=1
        s14_val = 4032
        s37_val = 1
        s46_val = 1

        s14 = torch.tensor(s14_val, dtype=torch.int64)
        s37 = torch.tensor(s37_val, dtype=torch.int64)
        s46 = torch.tensor(s46_val, dtype=torch.int64)

        # Expected result calculation (using Python integers to ensure correctness)
        # (24*1 + 672) * ((4032*1)//2016) + 21
        # = 696 * 2 + 21 = 1392 + 21 = 1413
        # 1413 // 22 = 64
        expected = torch.tensor(64, dtype=torch.int64)

        # Run eager mode to verify basic correctness
        eager_result = floor_div_expr(s14, s37, s46)
        self.assertEqual(eager_result, expected)

        # Run with torch.compile to trigger the inductor lowering
        # This is where the bug (FloorDiv -> Rational) would manifest
        compiled_func = torch.compile(floor_div_expr)
        compiled_result = compiled_func(s14, s37, s46)
        
        # Assert the compiled result matches the expected integer result
        self.assertEqual(compiled_result, expected)
        
        # Ensure the result is an integer tensor, not a float
        self.assertEqual(compiled_result.dtype, torch.int64)

    def test_floor_div_no_rational_conversion(self):
        """
        Specific check to ensure FloorDiv is not converted to Mul(Rational).
        We inspect the lowered code or behavior if possible, but primarily
        check the numerical correctness which would fail if converted to float.
        """
        # Use values where floor division differs from true division
        # (24*1 + 672) * ((1*1)//2016) + 21
        # = 696 * 0 + 21 = 21
        # 21 // 22 = 0
        # If converted to Rational: 21 * (1/22) = 0.954... -> float -> 0 (if cast back) or error
        
        s14 = torch.tensor(1, dtype=torch.int64)
        s37 = torch.tensor(1, dtype=torch.int64)
        s46 = torch.tensor(1, dtype=torch.int64)
        
        def simple_div(x, y, z):
            return torch.div((24 * y + 672) * ((x * z) // 2016) + 21, 22, rounding_mode='floor')

        expected = torch.tensor(0, dtype=torch.int64)
        
        compiled_simple = torch.compile(simple_div)
        result = compiled_simple(s14, s37, s46)
        
        self.assertEqual(result, expected)
        self.assertEqual(result.dtype, torch.int64)

if __name__ == "__main__":
    import unittest
    unittest.main()