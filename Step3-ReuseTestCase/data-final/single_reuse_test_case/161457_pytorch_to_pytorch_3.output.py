import torch
import torch.library
import unittest

class TestLibraryImplAbstract(unittest.TestCase):
    def test_custom_op_with_impl_abstract(self):
        # Define a custom operator
        torch.library.define("test_ns::my_op", "(Tensor x, Tensor y) -> Tensor")

        # Concrete implementation
        def my_op_impl(x, y):
            return x + y

        torch.library.impl("test_ns::my_op", my_op_impl, "CPU")
        if torch.cuda.is_available():
            torch.library.impl("test_ns::my_op", my_op_impl, "CUDA")

        # Abstract implementation (FakeTensor implementation)
        # This is the API under test: torch.library.impl_abstract
        # It defines the behavior for FakeTensors used during tracing/compilation
        def my_op_abstract(x, y):
            return x

        torch.library.impl_abstract("test_ns::my_op", my_op_abstract)

        # Test function using the custom operator
        def fn(x, y):
            return torch.ops.test_ns.my_op(x, y)

        # Use bfloat16 as per the original bug report context
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(2, 2, dtype=torch.bfloat16, device=device)
        y = torch.randn(2, 2, dtype=torch.bfloat16, device=device)

        # Eager execution
        res_eager = fn(x, y)

        # Compiled execution (relies on impl_abstract for tracing)
        fn_compiled = torch.compile(fn)
        res_compiled = fn_compiled(x, y)

        # Check correctness
        self.assertTrue(torch.allclose(res_eager, res_compiled, atol=1e-3))

if __name__ == "__main__":
    unittest.main()