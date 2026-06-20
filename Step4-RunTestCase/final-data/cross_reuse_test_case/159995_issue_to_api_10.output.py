import unittest
import torch
import os
from torch.utils.cpp_extension import load

class TestAotiCompileWithCondAndCustomOps(unittest.TestCase):
    """
    Test case for Issue 159995: torch._inductor.aoti_compile_and_package
    fails with CUDA kernels inside of a torch.cond.
    """
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # Setup for custom CUDA extension
        # This test assumes 'op.cu' exists in the same directory.
        # The content of op.cu should define 'add_one' and 'add_two' kernels.
        self.cu_source_path = os.path.join(os.path.dirname(__file__), "op.cu")
        
        if not os.path.exists(self.cu_source_path):
            self.skipTest(f"Required CUDA source file not found: {self.cu_source_path}")

        try:
            # Load the custom extension
            self.op = load(
                name="add_extension", 
                sources=[self.cu_source_path], 
                verbose=True
            )
            
            # Define the custom library
            torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
            torch.library.define("myops::add_two", "(Tensor x) -> Tensor")
            
            # Implement the ops using the loaded CUDA kernels
            torch.library.impl("myops::add_one", "CUDA", self.op.add_one)
            torch.library.impl("myops::add_two", "CUDA", self.op.add_two)

            # Register fake implementations for meta tensor generation
            @torch.library.register_fake("myops::add_one")
            def _(x): return torch.empty_like(x)

            @torch.library.register_fake("myops::add_two")
            def _(x): return torch.empty_like(x)
            
        except Exception as e:
            self.skipTest(f"Failed to setup custom ops: {e}")

    def test_aoti_cond_custom_cuda(self):
        """
        Reproduces the bug where AOTI compilation fails or produces incorrect results
        when torch.cond is used to select between custom CUDA kernels.
        """
        class CondModel(torch.nn.Module):
            def forward(self, x):
                # Select kernel based on batch size
                return torch.cond(
                    x.shape[0] < 5, 
                    torch.ops.myops.add_one, 
                    torch.ops.myops.add_two, 
                    (x,)
                )

        model = CondModel()
        
        # Configure dynamic shapes for export
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=128)}}
        
        # Export the model
        sample_input = torch.zeros(3, device="cuda")
        exported = torch.export.export(
            model, 
            (sample_input,), 
            dynamic_shapes=dynamic_shapes
        )
        
        # Compile and package the exported model
        package_path = "model_cond_custom.pt2"
        
        try:
            torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
            
            # Load the packaged model
            aoti_model = torch._inductor.aoti_load_package(package_path)
            
            # Verify execution for the branch where batch >= 5 (add_two)
            # Input: zeros(6), Expected: ones(6) * 2
            test_input = torch.zeros(6, device="cuda")
            result = aoti_model(test_input)
            expected = torch.ones_like(test_input) * 2
            
            self.assertTrue(torch.allclose(result, expected), 
                          "AOTI model output does not match expected for batch >= 5")
            
            # Verify execution for the branch where batch < 5 (add_one)
            # Input: zeros(3), Expected: ones(3)
            test_input_small = torch.zeros(3, device="cuda")
            result_small = aoti_model(test_input_small)
            expected_small = torch.ones_like(test_input_small)
            
            self.assertTrue(torch.allclose(result_small, expected_small), 
                          "AOTI model output does not match expected for batch < 5")

        finally:
            # Clean up generated package
            if os.path.exists(package_path):
                os.remove(package_path)

if __name__ == "__main__":
    unittest.main()