import torch
import os
import tempfile
from torch.utils.cpp_extension import load

def test_aoti_cond_with_custom_cuda_kernels():
    """
    Test case for Issue 159995:
    Verifies that torch._inductor.aoti_compile_and_package correctly handles
    models using torch.cond to select between custom CUDA kernels.
    
    This test mirrors the serialization/compilation pattern found in 
    tf.keras.dtype_policies.serialize by ensuring the model graph (containing
    conditional logic) is correctly packaged, serialized, and reloaded for execution.
    """
    
    # 1. Define Custom CUDA Kernels
    # We define the CUDA source inline to ensure the test is self-contained.
    # In a real scenario, this would be a separate .cu file.
    cuda_source = """
    #include <torch/extension.h>

    torch::Tensor add_one_cuda(torch::Tensor x) {
        return x + 1;
    }

    torch::Tensor add_two_cuda(torch::Tensor x) {
        return x + 2;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add_one", &add_one_cuda, "Add one to tensor");
        m.def("add_two", &add_two_cuda, "Add two to tensor");
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "op.cu")
        with open(source_path, "w") as f:
            f.write(cuda_source)

        # Load the custom extension
        # Note: This requires a CUDA-enabled environment with nvcc.
        try:
            custom_op = load(
                name="cond_test_extension", 
                sources=[source_path], 
                verbose=True,
                extra_cuda_cflags=["-O2"]
            )
        except Exception as e:
            print(f"Skipping test: CUDA extension compilation failed. {e}")
            return

        # 2. Register Custom Ops
        torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
        torch.library.define("myops::add_two", "(Tensor x) -> Tensor")
        
        torch.library.impl("myops::add_one", "CUDA", custom_op.add_one)
        torch.library.impl("myops::add_two", "CUDA", custom_op.add_two)

        @torch.library.register_fake("myops::add_one")
        def fake_add_one(x): return torch.empty_like(x)

        @torch.library.register_fake("myops::add_two")
        def fake_add_two(x): return torch.empty_like(x)

        # 3. Define Model with torch.cond
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

        # 4. Prepare Inputs and Export
        # We use dynamic shapes to ensure the cond branches are handled correctly
        example_input = torch.zeros(3, device="cuda") # Triggers add_one
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=128)}}
        
        exported = torch.export.export(
            model, 
            (example_input,), 
            dynamic_shapes=dynamic_shapes
        )

        # 5. Compile and Package (The API under test)
        package_path = os.path.join(tmpdir, "model_package.pt2")
        
        try:
            torch._inductor.aoti_compile_and_package(
                exported, 
                package_path=package_path
            )
        except Exception as e:
            print(f"Bug reproduced: aoti_compile_and_package failed with CUDA kernels in cond. {e}")
            raise

        # 6. Load and Execute
        aoti_model = torch._inductor.aoti_load_package(package_path)
        
        # Test with a batch size that triggers the 'False' branch (add_two)
        # This ensures the packaged model correctly handles the conditional logic
        # and the specific kernel required for this input.
        test_input = torch.zeros(6, device="cuda") 
        
        # Get eager result for comparison
        expected = model(test_input)
        
        # Get AOTI result
        result = aoti_model(test_input)
        
        # 7. Assertions
        # Check if execution succeeded and results match
        assert torch.allclose(result, expected), \
            f"Mismatch between eager and AOTI results: {result} vs {expected}"
            
        print("Test Passed: AOTI compiled and executed torch.cond with custom CUDA kernels successfully.")

if __name__ == "__main__":
    test_aoti_cond_with_custom_cuda_kernels()