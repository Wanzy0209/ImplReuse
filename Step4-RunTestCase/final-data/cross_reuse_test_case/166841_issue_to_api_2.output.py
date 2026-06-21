import torch
import torch.export
import torch._inductor
import os
import shutil

# --- 1. Model Definition ---
# This test case adapts the original bug reproduction logic to use 'add' 
# (semantically similar to tf.experimental.numpy.add) instead of 'scatter_add_'.
# It aims to verify if the incorrect CUDA kernel generation bug affects 
# standard element-wise operations in mixed-device (CPU/CUDA) AOT compilation.

class MyModelAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Buffers are on CPU, similar to the original bug report setup
        self.register_buffer(
            "cpu_a",
            torch.tensor([1.0, 2.0, 3.0, 4.0], device='cpu', dtype=torch.float32)
        )
        self.register_buffer(
            "cpu_b",
            torch.tensor([5.0, 6.0, 7.0, 8.0], device='cpu', dtype=torch.float32)
        )

    def forward(self, cuda_input):
        # Inputs are on CUDA

        # 1. Operation on CPU tensors using 'add' (similar to tf.experimental.numpy.add)
        # We perform the addition on CPU tensors.
        cpu_result = torch.add(self.cpu_a, self.cpu_b)

        # 2. Move result to CUDA and continue on CUDA
        # This forces the graph to contain both CPU and CUDA device types,
        # which is the trigger for the original bug.
        cuda_result = cpu_result.to('cuda') + cuda_input
        return cuda_result

# --- 2. Setup and Compile ---
model = MyModelAdd().eval()
cuda_input = torch.randn(4, device='cuda')
example_args = (cuda_input,)

print("Exporting model with CPU 'add' and CUDA input...")
ep = torch.export.export(model, example_args)

output_dir = "/tmp/test_aoti_add_model"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
package = os.path.join(output_dir, "model_package.pt2")

print("Starting AOTInductor compilation...")
try:
    torch._inductor.aoti_compile_and_package(
        ep,
        package_path=package,
    )
    print(f"Compiled package at: {package}")

    # --- 3. Load and Run ---
    print("\nAttempting to load and run compiled model...")
    loaded = torch._inductor.aoti_load_package(package)
    print("Model package loaded successfully.")

    result = loaded(*example_args)
    print("Model ran successfully with the loaded package.")

    # Verify correctness
    expected = model(*example_args)
    assert torch.allclose(result, expected), "Output mismatch between eager and AOTI run"
    print("Test passed: Output matches eager execution.")

except Exception as e:
    print(f"Test failed with error: {e}")
    # If the bug affects 'add' as well, we expect a runtime error here 
    # similar to "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu"
    raise