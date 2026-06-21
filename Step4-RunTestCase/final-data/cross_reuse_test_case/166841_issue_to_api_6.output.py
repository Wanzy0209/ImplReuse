import torch
import torch.export
import torch._inductor
import unittest
import tempfile
import os
import shutil

class TestAOTIScatterAddMultiDevice(unittest.TestCase):
    """
    Test case for Issue 166841:
    Verifies that aoti_compile_and_package correctly generates CPU kernels
    for scatter_add_ operations on CPU tensors, even when the model contains
    mixed CPU and CUDA operations.
    """

    def test_scatter_add_cpu_cuda_mixed(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # --- 1. Model Definition ---
        class MixedDeviceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Buffers are explicitly on CPU
                self.register_buffer(
                    "index",
                    torch.tensor([1, 4, 1, 7], device='cpu', dtype=torch.int64)
                )
                self.register_buffer(
                    "src",
                    torch.ones(4, device='cpu', dtype=torch.int64)
                )

            def forward(self, matrix, vector):
                # Inputs are on CUDA

                # 1. Operation on CPU tensors (scatter_add_)
                # This is the operation that was incorrectly generating a CUDA kernel
                z = torch.zeros((vector.shape[0],), device='cpu', dtype=torch.int64)
                z.scatter_add_(0, self.index, self.src)

                # 2. Move result to CUDA and continue on CUDA
                # This ensures the graph contains both device types
                v = vector + z.to(vector.dtype).to('cuda')
                return torch.matmul(matrix, v)

        # --- 2. Setup Inputs ---
        model = MixedDeviceModel().eval()
        matrix = torch.randn(10, 10, device='cuda')
        vector = torch.randn(10, device='cuda')
        example_args = (matrix, vector)

        # Get expected output from eager execution
        expected_output = model(*example_args)

        # --- 3. Export ---
        print("Exporting model...")
        ep = torch.export.export(model, example_args)

        # --- 4. Compile and Package ---
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = os.path.join(tmpdir, "model_package.pt2")
            
            print("Starting AOTInductor compilation...")
            # Bug location: This was generating aoti_torch_cuda_scatter_reduce_two_out
            # for the CPU scatter_add_ operation.
            torch._inductor.aoti_compile_and_package(
                ep,
                package_path=package_path,
            )
            print(f"Compiled package at: {package_path}")

            # --- 5. Load and Run ---
            print("\nAttempting to load and run compiled model...")
            # Bug location: This was failing with 
            # "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu"
            loaded_model = torch._inductor.aoti_load_package(package_path)
            print("Model package loaded successfully.")

            actual_output = loaded_model(*example_args)
            print("Model ran successfully with the loaded package.")

            # --- 6. Verification ---
            self.assertTrue(
                torch.allclose(expected_output, actual_output),
                "Output of compiled model does not match eager execution"
            )

if __name__ == '__main__':
    unittest.main()