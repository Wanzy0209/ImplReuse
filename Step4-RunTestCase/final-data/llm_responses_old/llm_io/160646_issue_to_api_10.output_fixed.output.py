import tempfile
import torch

# Check for CUDA availability as the original bug report relies on it
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    exit(0)

# Check for torch.export availability (requires PyTorch >= 2.1)
if not hasattr(torch, 'export'):
    print("torch.export is not available. This test requires PyTorch >= 2.1. Skipping test.")
    exit(0)

class BesselModule(torch.nn.Module):
    """
    A module that uses torch.special.bessel_j0 to test AOT compilation.
    This replaces the custom Triton op from the original bug report with 
    the similar API to verify the compilation pipeline.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Leveraging the similar API: torch.special.bessel_j0
        return torch.special.bessel_j0(x)

def main():
    model = BesselModule().to("cuda")
    
    # Prepare input data
    # Using a size consistent with the original bug report
    x = torch.randn(1024, device="cuda")
    
    # Export the model with dynamic shapes
    # Preserving the export logic from the original bug reproduction
    with torch.inference_mode():
        exported_model = torch.export.export(
            model, 
            (x,), 
            dynamic_shapes=({0: torch.export.Dim("dim")},)
        )
    
    # AOT Compile and Package
    # This is the specific step that failed in PyTorch 2.8 for Triton ops.
    # We test it here with torch.special.bessel_j0.
    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = tmpdir + "/package.pt2"
        
        try:
            torch._inductor.aoti_compile_and_package(
                exported_model,
                package_path=package_path,
            )
            print("Test Passed: AOT compilation successful for torch.special.bessel_j0.")
        except Exception as e:
            print(f"Test Failed: AOT compilation failed with error: {e}")
            raise

if __name__ == "__main__":
    main()