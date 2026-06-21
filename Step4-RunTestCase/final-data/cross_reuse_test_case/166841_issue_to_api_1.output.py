import torch
import torch.export
import torch._inductor
import os
import shutil

def test_aoti_mixed_device_scatter_add():
    """
    Test case for Issue 166841: aoti_compile_and_package incorrectly generates 
    CUDA kernel call for CPU scatter_add_ in multi-device models.
    
    This test incorporates the semantics of tf.math.add (via torch.add) to ensure
    mixed device operations are handled correctly during AOT compilation.
    """
    
    class MixedDeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Buffers are on CPU
            self.register_buffer(
                "index",
                torch.tensor([1, 4, 1, 7], device='cpu', dtype=torch.int64)
            )
            self.register_buffer(
                "src",
                torch.ones(4, device='cpu', dtype=torch.int64)
            )
            # Buffer for the 'add' operation (similar to tf.math.add)
            self.register_buffer(
                "addend",
                torch.tensor([2, 2, 2, 2], device='cpu', dtype=torch.int64)
            )

        def forward(self, matrix, vector):
            # Inputs are on CUDA

            # 1. Operation on CPU tensors
            z = torch.zeros((vector.shape[0],), device='cpu', dtype=torch.int64)
            
            # The bug trigger: scatter_add_ on CPU
            z.scatter_add_(0, self.index, self.src)

            # 2. Leverage similar API (tf.math.add -> torch.add)
            # Performing an addition on the CPU result to mix operations
            # This mirrors the context handling pattern seen in the similar API
            added_result = torch.add(z, self.addend)

            # 3. Move result to CUDA and continue on CUDA
            v = vector + added_result.to(vector.dtype).to('cuda')
            return torch.matmul(matrix, v)

    # --- Setup ---
    model = MixedDeviceModel().eval().to('cpu')
    matrix = torch.randn(10, 10, device='cuda')
    vector = torch.randn(10, device='cuda')
    example_args = (matrix, vector)

    # --- Export ---
    print("Exporting model...")
    ep = torch.export.export(model, example_args)

    output_dir = "/tmp/test_aoti_mixed_device"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    package = os.path.join(output_dir, "model_package.pt2")

    # --- Compile ---
    print("Starting AOTInductor compilation...")
    try:
        torch._inductor.aoti_compile_and_package(
            ep,
            package_path=package,
        )
    except Exception as e:
        print(f"Compilation failed: {e}")
        raise

    print(f"Compiled package at: {package}")

    # --- Load and Run ---
    print("\nAttempting to load and run compiled model...")
    try:
        loaded = torch._inductor.aoti_load_package(package)
        print("Model package loaded successfully.")

        result = loaded(*example_args)
        print("Model ran successfully with the loaded package.")
        
        # Basic assertion to ensure execution completed
        assert result is not None
        assert result.shape == (10, 10)
        
    except RuntimeError as e:
        if "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu" in str(e):
            print("BUG REPRODUCED: CUDA kernel called on CPU tensors.")
            raise
        else:
            raise

if __name__ == "__main__":
    test_aoti_mixed_device_scatter_add()