import torch
import torch._inductor
import os
import tempfile
from torch.utils.cpp_extension import load

def test_aoti_compile_and_package_with_cond_and_cuda():
    """
    Test case for Issue 159995:
    Verifies that torch._inductor.aoti_compile_and_package works correctly
    with CUDA kernels inside of torch.cond.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    # Define the CUDA kernel source code
    cuda_source = r"""
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    __global__ void add_one_kernel(float* x, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            x[idx] += 1.0f;
        }
    }

    __global__ void add_two_kernel(float* x, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            x[idx] += 2.0f;
        }
    }

    torch::Tensor add_one(torch::Tensor x) {
        auto size = x.numel();
        add_one_kernel<<<(size + 255) / 256, 256>>>(x.data_ptr<float>(), size);
        return x;
    }

    torch::Tensor add_two(torch::Tensor x) {
        auto size = x.numel();
        add_two_kernel<<<(size + 255) / 256, 256>>>(x.data_ptr<float>(), size);
        return x;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add_one", &add_one, "Add one");
        m.def("add_two", &add_two, "Add two");
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the CUDA source to a file
        op_cu_path = os.path.join(tmpdir, "op.cu")
        with open(op_cu_path, "w") as f:
            f.write(cuda_source)

        # Load the custom CUDA extension
        try:
            op = load(name="add_extension", sources=[op_cu_path], verbose=True)
        except Exception as e:
            # If compilation fails (e.g., no nvcc), we skip the test logic
            # but the structure remains valid for the test case.
            print(f"Failed to load CUDA extension: {e}")
            return

        # Define the custom ops
        torch.library.define("myops::add_one", "(Tensor x) -> Tensor")
        torch.library.define("myops::add_two", "(Tensor x) -> Tensor")
        torch.library.impl("myops::add_one", "CUDA", op.add_one)
        torch.library.impl("myops::add_two", "CUDA", op.add_two)

        # Register fake implementations for meta tensors
        @torch.library.register_fake("myops::add_one")
        def _(x): return torch.empty_like(x)

        @torch.library.register_fake("myops::add_two")
        def _(x): return torch.empty_like(x)

        # Define the model using torch.cond
        class M(torch.nn.Module):
            def forward(self, x):
                # The bug occurs specifically when torch.cond selects between custom CUDA kernels
                return torch.cond(x.shape[0] < 5, torch.ops.myops.add_one, torch.ops.myops.add_two, (x,))

        model = M()

        # Export the model with dynamic shapes
        # This matches the repro logic
        exported = torch.export.export(
            model, 
            (torch.zeros(3, device="cuda"),), 
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=128)}}
        )

        # Compile and package the model
        package_path = os.path.join(tmpdir, "model.pt2")
        torch._inductor.aoti_compile_and_package(exported, package_path=package_path)

        # Load the packaged model
        aoti_model = torch._inductor.aoti_load_package(package_path)

        # Test Case 1: Batch size < 5 (Should call add_one)
        input_1 = torch.zeros(3, device="cuda")
        result_1 = aoti_model(input_1)
        expected_1 = torch.ones(3, device="cuda")
        assert torch.allclose(result_1, expected_1), f"Test Case 1 Failed: Expected {expected_1}, got {result_1}"

        # Test Case 2: Batch size >= 5 (Should call add_two)
        input_2 = torch.zeros(6, device="cuda")
        result_2 = aoti_model(input_2)
        expected_2 = torch.full((6,), 2.0, device="cuda")
        assert torch.allclose(result_2, expected_2), f"Test Case 2 Failed: Expected {expected_2}, got {result_2}"

        print("All test cases passed.")

if __name__ == "__main__":
    test_aoti_compile_and_package_with_cond_and_cuda()