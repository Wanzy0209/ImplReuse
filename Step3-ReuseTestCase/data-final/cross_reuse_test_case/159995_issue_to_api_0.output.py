import torch
import torch._inductor
import os
import tempfile
from torch.utils.cpp_extension import load

def test_cond_with_custom_cuda_aoti():
    """
    Test case for Issue 159995:
    torch._inductor.aoti_compile_and_package fails with CUDA kernels inside of torch.cond.
    
    This test mirrors the pattern of serializing a module artifact (similar to 
    tf.keras.activations.serialize) by using torch._inductor.aoti_compile_and_package
    to save a model containing conditional logic with custom CUDA kernels.
    """
    
    # Setup: Create a minimal CUDA extension
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define a simple CUDA kernel source
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void add_kernel(float* x, float* y, int size, float val) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                y[idx] = x[idx] + val;
            }
        }

        torch::Tensor add_one(torch::Tensor x) {
            auto y = torch::empty_like(x);
            int size = x.numel();
            add_kernel<<<(size + 255)/256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), size, 1.0f);
            return y;
        }

        torch::Tensor add_two(torch::Tensor x) {
            auto y = torch::empty_like(x);
            int size = x.numel();
            add_kernel<<<(size + 255)/256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), size, 2.0f);
            return y;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("add_one", &add_one, "Add one");
            m.def("add_two", &add_two, "Add two");
        }
        """
        
        source_file = os.path.join(tmpdir, "custom_ops.cu")
        with open(source_file, "w") as f:
            f.write(cuda_source)

        # Load the custom ops
        try:
            custom_ops = load(name="custom_ops", sources=[source_file], verbose=False)
        except RuntimeError as e:
            print(f"Skipping test: Could not load CUDA extension. {e}")
            return

        # Register the custom ops
        torch.library.define("test::add_one", "(Tensor x) -> Tensor")
        torch.library.define("test::add_two", "(Tensor x) -> Tensor")
        
        torch.library.impl("test::add_one", "CUDA", custom_ops.add_one)
        torch.library.impl("test::add_two", "CUDA", custom_ops.add_two)

        @torch.library.register_fake("test::add_one")
        def fake_add_one(x): return torch.empty_like(x)

        @torch.library.register_fake("test::add_two")
        def fake_add_two(x): return torch.empty_like(x)

        # Define the model using torch.cond
        class CondModel(torch.nn.Module):
            def forward(self, x):
                # The bug occurs here: selecting between custom CUDA kernels
                return torch.cond(
                    x.shape[0] < 5, 
                    torch.ops.test.add_one, 
                    torch.ops.test.add_two, 
                    [x]
                )

        model = CondModel()
        
        # Export the model with dynamic shapes
        example_inputs = (torch.randn(3, device="cuda"),)
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=10)}}
        
        exported_program = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)

        # Leverage the pattern of the similar API (serialize) to package the model
        # Similar to: byte_str = stablehlo.serialize_portable_artifact_str(module_str, target)
        package_path = os.path.join(tmpdir, "cond_model.pt2")
        torch._inductor.aoti_compile_and_package(exported_program, package_path=package_path)

        # Load and verify the serialized artifact
        loaded_model = torch._inductor.aoti_load_package(package_path)

        # Test execution path 1: batch < 5 (add_one)
        input_1 = torch.zeros(3, device="cuda")
        output_1 = loaded_model(input_1)
        expected_1 = torch.ones(3, device="cuda")
        assert torch.allclose(output_1, expected_1), "Output mismatch for batch < 5"

        # Test execution path 2: batch >= 5 (add_two)
        input_2 = torch.zeros(6, device="cuda")
        output_2 = loaded_model(input_2)
        expected_2 = torch.ones(6, device="cuda") * 2
        assert torch.allclose(output_2, expected_2), "Output mismatch for batch >= 5"

        print("Test Passed: torch.cond with custom CUDA kernels works in AOTI.")

if __name__ == "__main__":
    test_cond_with_custom_cuda_aoti()