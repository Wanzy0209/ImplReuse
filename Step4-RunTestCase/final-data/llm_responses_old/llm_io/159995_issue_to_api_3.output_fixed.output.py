import torch
import os
import sys
import tempfile
from torch.utils.cpp_extension import load

# Check for torch._inductor availability to handle environments where it is not installed
try:
    import torch._inductor
except ImportError:
    print("Skipping test: torch._inductor is not available in this environment.")
    sys.exit(0)

# Helper function to wrap the compilation and loading logic, 
# inspired by the serialization pattern in the similar API.
def compile_and_load_module(exported_model, package_path):
    """
    Compiles the exported model into a package and loads it back.
    This mirrors the serialization pattern seen in the similar API.
    """
    torch._inductor.aoti_compile_and_package(exported_model, package_path=package_path)
    return torch._inductor.aoti_load_package(package_path)

def test_aoti_cond_with_custom_cuda_ops():
    # 1. Setup Custom CUDA Extension
    # We define the CUDA kernel source inline to ensure the test is self-contained.
    cuda_source = """
    #include <torch/extension.h>

    torch::Tensor add_one(torch::Tensor x) {
        return x + 1;
    }

    torch::Tensor add_two(torch::Tensor x) {
        return x + 2;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add_one", &add_one, "Add one");
        m.def("add_two", &add_two, "Add two");
    }
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "op.cu")
        with open(source_file, "w") as f:
            f.write(cuda_source)
            
        # Load the custom op
        try:
            op = load(
                name="add_extension", 
                sources=[source_file], 
                verbose=True, 
                extra_cuda_cflags=["-O2"]
            )
        except Exception as e:
            print(f"Skipping test: CUDA extension loading failed. {e}")
            return

        # 2. Define Custom Ops
        torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
        torch.library.define("myops::add_two", "(Tensor x) -> Tensor")
        torch.library.impl("myops::add_one", "CUDA", op.add_one)
        torch.library.impl("myops::add_two", "CUDA", op.add_two)

        @torch.library.register_fake("myops::add_one")
        def _(x): return torch.empty_like(x)

        @torch.library.register_fake("myops::add_two")
        def _(x): return torch.empty_like(x)

        # 3. Define Model with torch.cond
        class CondModel(torch.nn.Module):
            def forward(self, x):
                # torch.cond chooses between two CUDA kernels based on input shape
                return torch.cond(
                    x.shape[0] < 5, 
                    torch.ops.myops.add_one, 
                    torch.ops.myops.add_two, 
                    (x,)
                )

        model = CondModel()

        # 4. Export the Model
        # Using dynamic shapes to ensure the conditional logic is preserved
        example_inputs = (torch.zeros(3, device="cuda"),)
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=128)}}
        
        exported = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)

        # 5. Compile and Load using the helper function
        package_path = os.path.join(tmpdir, "model.pt2")
        aoti_model = compile_and_load_module(exported, package_path)

        # 6. Verify Execution
        # Case A: Batch size < 5 (should call add_one)
        input_small = torch.zeros(3, device="cuda")
        result_small = aoti_model(input_small)
        expected_small = torch.ones(3, device="cuda")
        assert torch.allclose(result_small, expected_small), \
            f"Failed for batch < 5: expected {expected_small}, got {result_small}"

        # Case B: Batch size >= 5 (should call add_two)
        # This is the specific case mentioned in the bug report (zeros(6))
        input_large = torch.zeros(6, device="cuda")
        result_large = aoti_model(input_large)
        expected_large = torch.ones(6, device="cuda") * 2
        assert torch.allclose(result_large, expected_large), \
            f"Failed for batch >= 5: expected {expected_large}, got {result_large}"

        print("Test passed: torch.cond with CUDA kernels works correctly in AOTI.")

if __name__ == "__main__":
    test_aoti_cond_with_custom_cuda_ops()