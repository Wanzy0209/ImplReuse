import tempfile
import torch
import torch.special
from torch import nn

# Define a module using the similar API: torch.special.scaled_modified_bessel_k0
# This replaces the custom Triton op from the original bug report.
class BesselModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Using the similar API to test the AOT compilation pipeline
        return torch.special.scaled_modified_bessel_k0(x)

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    model = BesselModule().to("cuda")
    
    # Create input data
    # scaled_modified_bessel_k0 works with real numbers; we use positive values for stability
    x = torch.randn(1024, device="cuda").abs() + 0.1

    # Export the model (mirroring the bug report's export step)
    with torch.inference_mode():
        exported_model = torch.export.export(
            model, 
            (x,), 
            dynamic_shapes=({0: torch.export.Dim("dim")},)
        )

    # Attempt AOT compilation (mirroring the bug report's failure point)
    # This verifies that the AOT compiler handles the similar API correctly.
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = tmpdir + "/package.pt2"
        try:
            torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=package_path,
            )
            print("AOT compilation with torch.special.scaled_modified_bessel_k0 succeeded.")
        except Exception as e:
            print(f"AOT compilation failed: {e}")
            raise

if __name__ == "__main__":
    main()