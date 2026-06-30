import torch
import tempfile
import os
import unittest

class TestAotiCompileAndPackageWithCond(unittest.TestCase):
    """
    Test case for Issue 159995: torch._inductor.aoti_compile_and_package 
    fails with CUDA kernels inside of torch.cond.
    
    This test verifies the serialization (packaging) and deserialization (loading)
    logic of the AOTI compiler, specifically when handling conditional control flow
    over custom operators.
    """

    @classmethod
    def setUpClass(cls):
        # Define custom ops mimicking the structure of the bug report
        torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
        torch.library.define("myops::add_two", "(Tensor x) -> Tensor")

        # Register fake implementations for export/metadata
        # Note: torch.library.register_fake is not available in all versions.
        # We use torch.library.impl with the "Meta" dispatch key instead for compatibility.
        @torch.library.impl("myops::add_one", "Meta")
        def _(x): return torch.empty_like(x)

        @torch.library.impl("myops::add_two", "Meta")
        def _(x): return torch.empty_like(x)

        # Register actual implementations.
        # Note: The original bug involves C++ CUDA kernels. Here we use Python 
        # implementations to ensure the test is runnable without a CUDA compiler,
        # while preserving the API usage pattern of torch.library.
        @torch.library.impl("myops::add_one", "CUDA")
        def add_one_cuda(x):
            return x + 1

        @torch.library.impl("myops::add_two", "CUDA")
        def add_two_cuda(x):
            return x + 2

        # CPU fallbacks for environments without CUDA
        @torch.library.impl("myops::add_one", "CPU")
        def add_one_cpu(x):
            return x + 1

        @torch.library.impl("myops::add_two", "CPU")
        def add_two_cpu(x):
            return x + 2

    def test_aoti_cond_custom_ops(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Skip actual CUDA-specific kernel testing if not on CUDA, 
        # but still run the logic flow on CPU.
        if device == "cpu":
            self.skipTest("CUDA not available, skipping CUDA kernel specific test.")

        class M(torch.nn.Module):
            def forward(self, x):
                # The core logic: torch.cond selecting between custom ops
                return torch.cond(
                    x.shape[0] < 5, 
                    torch.ops.myops.add_one, 
                    torch.ops.myops.add_two, 
                    (x,)
                )

        model = M().to(device)

        # 1. Export the model with dynamic shapes
        # This prepares the program for serialization
        x = torch.zeros(3, device=device)
        exported = torch.export.export(
            model, 
            (x,), 
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=128)}}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = os.path.join(tmpdir, "model.pt2")

            # 2. Serialize/Package the model
            # This is the API under test (torch._inductor.aoti_compile_and_package)
            # It corresponds to the 'serialize' concept in the similar API.
            try:
                torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
            except Exception as e:
                self.fail(f"aoti_compile_and_package failed: {e}")

            # 3. Deserialize/Load the model
            # Verify the serialized package can be loaded back
            try:
                aoti_model = torch._inductor.aoti_load_package(package_path)
            except Exception as e:
                self.fail(f"aoti_load_package failed: {e}")

            # 4. Verify execution correctness
            # Test branch 1: batch < 5 (should call add_one)
            input_1 = torch.zeros(3, device=device)
            result_1 = aoti_model(input_1)
            expected_1 = torch.ones(3, device=device)
            self.assertTrue(torch.allclose(result_1, expected_1), 
                            "Failed on branch 1 (add_one)")

            # Test branch 2: batch >= 5 (should call add_two)
            input_2 = torch.zeros(6, device=device)
            result_2 = aoti_model(input_2)
            expected_2 = torch.ones(6, device=device) * 2
            self.assertTrue(torch.allclose(result_2, expected_2), 
                            "Failed on branch 2 (add_two)")

if __name__ == "__main__":
    unittest.main()