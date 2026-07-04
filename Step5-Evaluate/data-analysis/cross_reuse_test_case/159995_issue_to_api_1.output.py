import torch
import os
import tempfile
from torch.testing import assert_close

def setup_custom_ops():
    """
    Defines the torch library ops using Python implementations to avoid
    compilation dependencies (Ninja/CUDA compiler) in the test environment.
    """
    # Define the custom ops
    torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
    torch.library.define("myops::add_two", "(Tensor x) -> Tensor")
    
    # Implement the ops for CUDA using Python functions
    # This mimics the behavior of the C++ kernels without requiring compilation.
    def add_one_impl(x):
        return x + 1.0

    def add_two_impl(x):
        return x + 2.0
        
    torch.library.impl("myops::add_one", "CUDA", add_one_impl)
    torch.library.impl("myops::add_two", "CUDA", add_two_impl)
    
    # Register fake implementations for meta tensor
    @torch.library.register_fake("myops::add_one")
    def _(x): return torch.empty_like(x)
    
    @torch.library.register_fake("myops::add_two")
    def _(x): return torch.empty_like(x)

def compile_and_execute_cond_model(model, args, dynamic_shapes, package_path):
    """
    Compiles and packages the model using AOTInductor, then loads and executes it.
    This function mirrors the structure of the similar API 'serialize' by taking
    the model definition, processing it (export/compile), and returning the result.
    """
    # 1. Export the model
    exported = torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
    
    # 2. Compile and Package
    torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
    
    # 3. Load and Execute
    aoti_model = torch._inductor.aoti_load_package(package_path)
    result = aoti_model(*args)
    
    return result

def test_aoti_cond_with_custom_cuda_kernels():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    setup_custom_ops()

    class M(torch.nn.Module):
        def forward(self, x):
            # Use torch.cond to switch between custom CUDA kernels
            return torch.cond(
                x.shape[0] < 5, 
                torch.ops.myops.add_one, 
                torch.ops.myops.add_two, 
                (x,)
            )

    model = M().cuda()

    # Define dynamic shapes for export
    dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=128)}}
    
    # Create a temporary path for the package
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = os.path.join(tmpdir, "model.pt2")
        
        # Input for export and execution
        # We test with a batch size of 6, which triggers add_two (the other branch)
        # This verifies that the conditional logic and custom kernels are packaged correctly.
        test_args = (torch.ones(6, device="cuda"),)
        
        # Run the compile and execute flow
        result = compile_and_execute_cond_model(
            model, 
            test_args, 
            dynamic_shapes, 
            package_path
        )
        
        # Since we passed ones(6) and shape[0] is 6 (>= 5), add_two is called.
        # Expected result: 1.0 + 2.0 = 3.0
        expected = torch.ones(6, device="cuda") + 2.0
        
        assert_close(result, expected)
        print("Test passed! AOTI compiled model with torch.cond and custom CUDA kernels executed correctly.")

if __name__ == "__main__":
    test_aoti_cond_with_custom_cuda_kernels()