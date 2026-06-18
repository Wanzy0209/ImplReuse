import torch
import torch.export
import torch._inductor
import os
import shutil

# --- 1. Minimal Model Definition ---
class MyModel(torch.nn.Module):
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

    def forward(self, matrix, vector):
        # Inputs are on CUDA

        # 1. Operation on CPU tensors
        z = torch.zeros((vector.shape[0],), device='cpu', dtype=torch.int64)
        scatter_result = z.scatter_add(0, self.index, self.src)

        # 2. Move result to CUDA and continue on CUDA
        v = vector + scatter_result.to(vector.dtype).to('cuda')
        return torch.matmul(matrix, v)

# --- 2. Setup and Compile ---
model = MyModel().eval().to('cpu')
matrix = torch.randn(10, 10, device='cuda')
vector = torch.randn(10, device='cuda')
example_args = (matrix, vector)

print("Exporting model...")
ep = torch.export.export(model, example_args)

output_dir = "/tmp/my_aoti_model"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
package = os.path.join(output_dir, "model_package.pt2")

print("Starting AOTInductor compilation...")
torch._inductor.aoti_compile_and_package(
    ep,
    package_path=package,
)
print(f"Compiled package at: {package}")

# --- 3. Load and Run (This is where it fails) ---
print("\nAttempting to load and run compiled model...")
loaded = torch._inductor.aoti_load_package(package)
print("\nModel package loaded successfully.")

loaded(*example_args)
print("Model ran successfully with the loaded package.")